from argparse import ArgumentParser, Namespace
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from train.flipoutseq2seqbart import FlipoutSeq2SeqBart
from src.train.trainconfig import MODEL_CONFIGS


@dataclass
class GenConfig:
    """Configuration for generating answers based on parsed arguments."""

    model: str
    method: str
    dataset: str
    n_reps: int
    batch_size: int                     # how many questions to process per time
    model_path: Path = field(init=False)
    test_df_path: Path = field(init=False)
    answers_dest_path: Path = field(init=False)

    def __post_init__(self) -> None:
        """Set some directories."""
        self.model_path = Path("models") / f"{self.dataset}-{self.model}-{self.method}"
        self.test_df_path = Path("data") / self.dataset / f"{self.dataset}-test.jsonl"
        self.answers_dest_path = Path("results") / self.dataset


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=["nq", "webquestions", "triviaqa"], required=True)
    parser.add_argument("--model", type=str, choices=["bart-large", "t5-large-ssm", "flan-t5-large"], required=True)
    parser.add_argument("--method", type=str, choices=["vanilla", "flipout"], required=True)
    parser.add_argument("--n_reps", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    return parser.parse_args()


def load_model(method_type: str, path: Path, device: torch.device) -> AutoModelForSeq2SeqLM:
    if method_type == "vanilla":
        return AutoModelForSeq2SeqLM.from_pretrained(path).train().to(device)
    if method_type == "flipout":
        return FlipoutSeq2SeqBart.from_pretrained(path).eval().to(device)
    msg = f"Unknown model type: {method_type}"
    raise ValueError(msg)


def generate_predictions_batched(model: AutoModelForSeq2SeqLM, tokenizer: AutoTokenizer, questions: list[str],
                                 device: torch.device, n_reps: int, batch_size: int) -> list[list[str]]:
    """Generate predictions in batches."""
    all_predictions = []

    for idx in tqdm(range(0, len(questions), batch_size)):
        chunk_qs = questions[idx:idx + batch_size]
        tok_qs = tokenizer(                                 # tokenize each question only once
            chunk_qs, max_length=32, truncation=True, padding="max_length", return_tensors="pt").to(device)

        batch_preds = []
        with torch.no_grad():
            for _ in range(n_reps):
                out_ids = model.generate(**tok_qs)
                preds = tokenizer.batch_decode(out_ids, skip_special_tokens=True)
                batch_preds.append(preds)                   # X_ij is i-th answer to j-th question

        transposed = [list(sample_preds) for sample_preds in zip(*batch_preds, strict=True)]
        all_predictions.extend(transposed)                  # X_ij is j-th answer to i-th question

    return all_predictions


def main() -> None:
    """Driver to make many forward passes for a stochastic model."""
    config = GenConfig(**vars(parse_args()))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(config.method, config.model_path, device)
    tokenizer = AutoTokenizer.from_pretrained(config.model_path)
    test_df = pd.read_json(config.test_df_path, lines=True)

    prefix = MODEL_CONFIGS[config.model]["prefix"]
    questions = test_df["question"].apply(lambda q: prefix + q).tolist()

    test_df["predictions"] = generate_predictions_batched(
        model, tokenizer, questions, device, config.n_reps, config.batch_size)

    Path.mkdir(config.answers_dest_path, parents=True, exist_ok=True)
    results_path = config.answers_dest_path / f"{config.method}-large.jsonl"
    test_df.to_json(results_path, orient="records", lines=True)


if __name__ == "__main__":
    main()
