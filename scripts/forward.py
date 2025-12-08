from argparse import ArgumentParser, Namespace
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from transformers import BartForConditionalGeneration, BartTokenizer

from src.train.flipoutbart import FlipoutBart


@dataclass
class GenConfig:
    """Configuration for generating answers based on parsed arguments."""

    model: str
    dataset: str
    n_reps: int
    batch_size: int
    model_path: Path = field(init=False)
    test_df_path: Path = field(init=False)
    answers_dest_path: Path = field(init=False)

    def __post_init__(self) -> None:
        """Set some directories."""
        self.model_path = Path("models") / f"{self.dataset}-{self.model}-large"
        self.test_df_path = Path("data") / self.dataset / f"{self.dataset}-test.jsonl"
        self.dest_path = Path("results") / self.dataset


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, choices=["mcdropout", "flipout"], required=True)
    parser.add_argument("--dataset", type=str, choices=["nq", "webquestions", "triviaqa"], required=True)
    parser.add_argument("--n_reps", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    return parser.parse_args()


def load_model(model_type: str, path: Path, device: torch.device) -> BartForConditionalGeneration:
    if model_type == "mcdropout":
        return BartForConditionalGeneration.from_pretrained(path).train().to(device)
    if model_type == "flipout":
        return FlipoutBart.from_pretrained(path).eval().to(device)
    msg = f"Unknown model type: {model_type}"
    raise ValueError(msg)


def generate_predictions_batched(model: BartForConditionalGeneration, tokenizer: BartTokenizer, questions: list[str],
                                 device: torch.device, n_reps: int, batch_size: int) -> list[list[str]]:
    """Generate predictions in batches."""
    all_predictions = []

    for idx in tqdm(range(0, len(questions), batch_size)):
        chunk_qs = questions[idx:idx + batch_size]
        tok_qs = tokenizer(                             # tokenize each question only once
            chunk_qs, max_length=32, truncation=True, padding="max_length", return_tensors="pt").to(device)

        batch_preds = []
        with torch.no_grad():
            for _ in range(n_reps):
                out_ids = model.generate(
                    **tok_qs, max_new_tokens=32, num_beams=1, do_sample=False, early_stopping=False)
                preds = tokenizer.batch_decode(out_ids, skip_special_tokens=True)
                batch_preds.append(preds)                   # X_ij is i-th answer to j-th question

        transposed = [list(sample_preds) for sample_preds in zip(*batch_preds, strict=True)]
        all_predictions.extend(transposed)                  # X_ij is j-th answer to i-th question

    return all_predictions


def main() -> None:
    """Driver to make many forward passes for a stochastic model."""
    config = GenConfig(**vars(parse_args()))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(config.model, config.model_path, device)
    tokenizer = BartTokenizer.from_pretrained(config.model_path)
    test_df = pd.read_json(config.test_df_path, lines=True)

    test_df["predictions"] = generate_predictions_batched(
        model, tokenizer, test_df["question"].tolist(), device, config.n_reps, config.batch_size)

    Path.mkdir(config.dest_path, parents=True, exist_ok=True)
    test_df.to_json(config.dest_path / f"{config.model}-large.jsonl", orient="records", lines=True)


if __name__ == "__main__":
    main()
