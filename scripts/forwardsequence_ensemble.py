from argparse import ArgumentParser, BooleanOptionalAction, Namespace
from pathlib import Path

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
    parser.add_argument("--batch_size", type=int, default=64)
    return parser.parse_args()


def generate_predictions_batched(models: list[AutoModelForSeq2SeqLM], tokenizer: AutoTokenizer, questions: list[str],
                                 device: torch.device, batch_size: int) -> list[list[str]]:
    """Generate predictions in batches."""
    all_predictions = []

    for idx in tqdm(range(0, len(questions), batch_size)):
        chunk_qs = questions[idx:idx + batch_size]
        tok_qs = tokenizer(                                 # tokenize each question only once
            chunk_qs, max_length=32, truncation=True, padding="max_length", return_tensors="pt").to(device)

        batch_preds = []
        with torch.no_grad():
            for model in models:
                out_ids = model.generate(**tok_qs)
                preds = tokenizer.batch_decode(out_ids, skip_special_tokens=True)
                batch_preds.append(preds)                   # X_ij is i-th answer to j-th question

        transposed = [list(sample_preds) for sample_preds in zip(*batch_preds, strict=True)]
        all_predictions.extend(transposed)                  # X_ij is j-th answer to i-th question

    return all_predictions


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
    prefix = MODEL_CONFIGS[args.model]["prefix"]
    questions = test_df["question"].apply(lambda q: prefix + q).tolist()

    test_df["predictions"] = generate_predictions_batched(models, tokenizer, questions, device, args.batch_size)

    results_path = Path("results") / args.dataset / f"ensemble{args.n_ensemble}-{suffix}.jsonl"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    test_df.to_json(results_path, orient="records", lines=True)


if __name__ == "__main__":
    main()
