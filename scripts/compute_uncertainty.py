from argparse import ArgumentParser, BooleanOptionalAction, Namespace
from collections.abc import Callable
from pathlib import Path

import pandas as pd

from src.eval.f1_rms import f1_rms_uncertainty
from src.eval.sbertdemo import frob
from src.eval.utils import variation_ratio, vote_entropy


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=["webquestions", "nq"], required=True)
    parser.add_argument("--method", type=str, choices=["mcdropout", "flipout", "ensemble5"], required=True)
    parser.add_argument(
        "--uq_method", type=str, choices=["f1", "bertscore", "variationratio", "voteentropy"], required=True)
    parser.add_argument("--use_soft", action=BooleanOptionalAction, required=True)
    return parser.parse_args()


def choose_method(uq_method: str) -> Callable[[list[str]], float]:
    """Determine UQ method based on user input."""
    match uq_method:
        case "f1":
            return f1_rms_uncertainty
        case "bertscore":
            return frob
        case "variationratio":
            return variation_ratio
        case "voteentropy":
            return vote_entropy
        case _:
            msg = f"Unexpected argument: {uq_method}"
            raise ValueError(msg)


def main() -> None:
    """Compute uncertainty scores based on answers from stochastic model."""
    args = parse_args()
    suffix = "soft" if args.use_soft else "hard"

    answer_path = Path("results") / args.dataset / f"{args.method}-{suffix}.jsonl"
    answers_df = pd.read_json(answer_path, lines=True)
    answers_df[args.uq_method] = answers_df["predictions"].apply(choose_method(args.uq_method))

    answer_dest_path = Path("results") / args.dataset / f"{args.method}-{suffix}-{args.uq_method}.jsonl"
    answers_df.to_json(answer_dest_path, orient="records", lines=True)


if __name__ == "__main__":
    main()
