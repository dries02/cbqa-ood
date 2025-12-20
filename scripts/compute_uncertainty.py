from argparse import ArgumentParser, Namespace
from collections.abc import Callable
from pathlib import Path

import pandas as pd

from src.eval.f1_rms import f1_rms_uncertainty
from src.eval.sbertdemo import frob


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, choices=["mcdropout", "flipout"], required=True)
    parser.add_argument("--dataset", type=str, choices=["nq", "webquestions", "triviaqa"], required=True)
    parser.add_argument("--method", type=str, choices=["f1", "bertscore"], required=True)
    return parser.parse_args()


def choose_method(method: str) -> Callable[[list[str]], float]:
    """Determine UQ method based on user input."""
    if method == "f1":
        return f1_rms_uncertainty
    if method == "bertscore":
        return frob
    msg = f"Unexpected argument: {method}"
    raise ValueError(msg)


def main() -> None:
    """Compute uncertainty scores based on answers from stochastic model."""
    args = parse_args()
    answer_path = Path("results") / args.dataset / f"{args.model}-large.jsonl"
    answers_df = pd.read_json(answer_path, lines=True)
    answers_df[args.method] = answers_df["predictions"].apply(choose_method(args.method))
    answer_dest_path = Path("results") / args.dataset / f"{args.model}-large-{args.method}.jsonl"
    answers_df.to_json(answer_dest_path, orient="records", lines=True)


if __name__ == "__main__":
    main()
