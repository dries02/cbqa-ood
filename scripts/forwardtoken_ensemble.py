from argparse import ArgumentParser, BooleanOptionalAction, Namespace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from src.train.trainconfig import MODEL_CONFIGS

MAX_Q_LEN = 32
MAX_ANS_LEN = 32


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=["webquestions", "nq"], required=True)
    parser.add_argument("--model", type=str, choices=["t5-large-ssm"], required=True)
    parser.add_argument("--use_soft", action=BooleanOptionalAction, required=True)
    parser.add_argument("--n_ensemble", type=int, default=5)
    return parser.parse_args()


@torch.no_grad()
def generate_answer_ensemble(
        models: list[AutoModelForSeq2SeqLM],
        tokenizer: AutoTokenizer,
        question: str,
        device: torch.device) -> dict:

    tok_q = tokenizer(
        question, max_length=MAX_Q_LEN, truncation=True, padding="max_length", return_tensors="pt").to(device)

    decoder_input_ids = torch.tensor([[models[0].config.decoder_start_token_id]], device=device)

    token_entropies = []
    token_mis = []

    for _ in range(MAX_ANS_LEN):
        all_probs = []

        for model in models:
            logits = model(**tok_q, decoder_input_ids=decoder_input_ids).logits[:, -1, :]  # (1, vocab)
            probs = torch.softmax(logits, dim=-1).squeeze(0)  # (vocab,)
            all_probs.append(probs)

        all_probs = torch.stack(all_probs)  # (n_ensemble, vocab)

        # Total entropy: H[E[p]]
        mean_probs = all_probs.mean(dim=0)
        entropy = -(mean_probs * torch.log(mean_probs + 1e-10)).sum()

        # Expected entropy: E[H[p]]
        per_model_entropy = -(all_probs * torch.log(all_probs + 1e-10)).sum(dim=-1)

        # MI = Total - Expected
        mutual_info = entropy - per_model_entropy.mean()

        token_entropies.append(entropy.item())
        token_mis.append(mutual_info.item())

        # Next token from mean probs
        next_token = mean_probs.argmax()
        decoder_input_ids = torch.cat([decoder_input_ids, next_token.view(1, 1)], dim=1)

        if next_token == tokenizer.eos_token_id:
            break

    return {
        "prediction": tokenizer.decode(decoder_input_ids[0], skip_special_tokens=True),
        "mean_mi": np.mean(token_mis),
        "max_mi": np.max(token_mis),
        "mean_entropy": np.mean(token_entropies),
        "max_entropy": np.max(token_entropies),
    }


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    suffix = "soft" if args.use_soft else "hard"

    model_paths = [
        Path("models") / f"{args.dataset}-{args.model}-mcdropout-{suffix}-{i}"
        for i in range(args.n_ensemble)
    ]

    for path in model_paths:
        if not path.exists():
            msg = f"Model not found: {path}"
            raise FileNotFoundError(msg)

    print(f"Loading {args.n_ensemble} models...")
    models = [
        AutoModelForSeq2SeqLM.from_pretrained(path).eval().to(device)
        for path in model_paths
    ]
    print(f"Models loaded. GPU memory: {torch.cuda.memory_allocated()/1e9:.1f}GB")

    tokenizer = AutoTokenizer.from_pretrained(model_paths[0])
    test_df = pd.read_json(f"data/{args.dataset}/{args.dataset}-test.jsonl", lines=True)
    prefix = MODEL_CONFIGS[args.model]["prefix"]

    tqdm.pandas()
    test_df[["prediction", "mean_mi", "max_mi", "mean_entropy", "max_entropy"]] = test_df["question"].progress_apply(
        lambda q: pd.Series(generate_answer_ensemble(models, tokenizer, prefix + q, device)))

    results_path = Path("results") / args.dataset / f"ensemble{args.n_ensemble}-{suffix}-token.jsonl"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    test_df.to_json(results_path, orient="records", lines=True)


if __name__ == "__main__":
    main()
