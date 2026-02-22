from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from src.train.trainconfig import MODEL_CONFIGS

MAX_Q_LEN = 32
MAX_ANS_LEN = 32

from dataclasses import dataclass
@dataclass
class Args:
    dataset: str
    model = "t5-large-ssm"
    use_soft = False
    n_ensemble = 5


@torch.no_grad()
def generate_answer_ensemble(
        models: list[AutoModelForSeq2SeqLM],
        tokenizer: AutoTokenizer,
        question: str,
        device: torch.device) -> dict:

    tok_q = tokenizer(
        question, max_length=MAX_Q_LEN, truncation=True, padding="max_length", return_tensors="pt").to(device)

    decoder_input_ids = torch.tensor([[models[0].config.decoder_start_token_id]], device=device)

    print()
    print(question)

    for _ in range(MAX_ANS_LEN):
        all_probs = []

        for model in models:
            logits = model(**tok_q, decoder_input_ids=decoder_input_ids).logits[:, -1, :]  # (1, vocab)
            probs = torch.softmax(logits, dim=-1).squeeze(0)  # (vocab,)
            all_probs.append(probs)

        all_probs = torch.stack(all_probs)  # (n_ensemble, vocab)

        # Total entropy: H[E[p]]
        mean_probs = all_probs.mean(dim=0)
        total = -(mean_probs * torch.log(mean_probs + 1e-10)).sum()

        # Expected entropy: E[H[p]]
        per_model_entropy = -(all_probs * torch.log(all_probs + 1e-10)).sum(dim=-1)
        aleatoric = per_model_entropy.mean()

        # MI := Total - Expected
        epistemic = total - aleatoric

        # Next token from mean probs
        next_token = mean_probs.argmax()

        k = 3
        topk_vals, topk_ids = torch.topk(mean_probs, k)
        for val, idx in zip(topk_vals, topk_ids, strict=True):
            print(f"{tokenizer.decode(idx)} with probability {val}")

        print(f"predicting: {tokenizer.decode(next_token)}({next_token.item()})"
              f" with TU {total.item():.3f}, EU {epistemic.item():.3f}, AU {aleatoric.item():.3f}")


        decoder_input_ids = torch.cat([decoder_input_ids, next_token.view(1, 1)], dim=1)

        if next_token == tokenizer.eos_token_id:
            break


def main() -> None:
    dataset = input("Please specify dataset (nq/webquestions) ")
    args = Args(dataset)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    suffix = "soft" if args.use_soft else "hard"

    model_paths = [
        Path("models") / f"{args.dataset}-{args.model}-mcdropout-{suffix}-0.1-{i}"
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
    # test_df = pd.read_json(f"data/{args.dataset}/{args.dataset}-test.jsonl", lines=True).head()
    prefix = MODEL_CONFIGS[args.model]["prefix"]

    tqdm.pandas()

    while True:
        question = input("? ")
        if question == "q":
            break

        generate_answer_ensemble(models, tokenizer, prefix + question, device)
    # test_df["question"].progress_apply(lambda q: pd.Series(generate_answer_ensemble(models, tokenizer, prefix + q, device)))


if __name__ == "__main__":
    main()
