from argparse import ArgumentParser, Namespace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from src.train.flipoutbart import FlipoutBart
from src.train.trainconfig import MODEL_CONFIGS

MAX_Q_LEN = 32
MAX_ANS_LEN = 32


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=["nq", "webquestions", "triviaqa"], required=True)
    parser.add_argument("--model", type=str, choices=["bart-large", "t5-large-ssm", "flan-t5-large"], required=True)
    parser.add_argument("--method", type=str, choices=["vanilla", "flipout"], required=True)
    parser.add_argument("--n_reps", type=int, default=30)
    return parser.parse_args()


def load_model(method_type: str, path: Path, device: torch.device) -> AutoModelForSeq2SeqLM:
    if method_type == "vanilla":
        return AutoModelForSeq2SeqLM.from_pretrained(path).train().to(device)
    if method_type == "flipout":
        return FlipoutBart.from_pretrained(path).eval().to(device)
    msg = f"Unknown model type: {method_type}"
    raise ValueError(msg)


@torch.no_grad()
def generate_answer(
        model: AutoModelForSeq2SeqLM, tokenizer: AutoTokenizer, question: str, device: torch.device,
        n_reps: int) -> dict:
    tok_q = tokenizer(
        question, max_length=MAX_Q_LEN, truncation=True, padding="max_length", return_tensors="pt").to(device)
    decoder_input_ids = torch.tensor([[model.config.decoder_start_token_id]], device=device)

    tok_q_batched = {k: v.repeat(n_reps, 1) for k, v in tok_q.items()}

    token_mis = []
    token_entropies = []
    for _ in range(MAX_ANS_LEN):
        decoder_batched = decoder_input_ids.repeat(n_reps, 1)  # (n_reps, seq_len)
        all_logits = model(**tok_q_batched, decoder_input_ids=decoder_batched).logits[:, -1, :]  # (n_reps, vocab)

        probs = torch.softmax(all_logits, dim=-1)
        mean_probs = probs.mean(dim=0)

        entropy = -(mean_probs * torch.log(mean_probs + 1e-10)).sum()
        per_pass_entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)
        mutual_info = entropy - per_pass_entropy.mean()
        token_mis.append(mutual_info.item())
        token_entropies.append(entropy.item())
        next_token = mean_probs.argmax()
        if next_token == tokenizer.eos_token_id:
            break

        decoder_input_ids = torch.cat([decoder_input_ids, next_token.view(1, 1)], dim=1)

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
    model_path = Path("models") / f"{args.dataset}-{args.model}-{args.method}-large"
    model = load_model(args.method, model_path, device)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    test_df = pd.read_json(f"data/{args.dataset}/{args.dataset}-test.jsonl", lines=True)

    prefix = MODEL_CONFIGS[args.model]["prefix"]
    tqdm.pandas()
    test_df[["prediction", "mean_mi", "max_mi", "mean_entropy", "max_entropy"]] = test_df["question"].progress_apply(
        lambda q: pd.Series(generate_answer(model, tokenizer, prefix + q, device, args.n_reps)))

    results_path = Path("results") / args.dataset / f"{args.method}-large-token.jsonl"
    test_df.to_json(results_path, orient="records", lines=True)


if __name__ == "__main__":
    main()
