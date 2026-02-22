from argparse import ArgumentParser, Namespace
from pathlib import Path

import pandas as pd


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=("webquestions", "nq"), required=True)
    parser.add_argument("--frac", type=float, choices=(0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0), required=True)
    return parser.parse_args()


def main() -> None:
    """Subsample the training set (without replacement) for analyzing disentanglement."""
    args = parse_args()
    source_path = Path("data") / args.dataset / f"{args.dataset}-train.jsonl"
    df = pd.read_json(source_path, lines=True)
    df_shuffled = df.sample(frac=1.0, replace=False, random_state=67)           # shuffle once, reproducible
    subset = df_shuffled.iloc[:int(args.frac * len(df_shuffled))]

    print(f"reduced {len(df)} to {len(subset)}")
    print(subset.head())
    dest_path = Path("data") / args.dataset / f"{args.dataset}-train-{args.frac}.jsonl"
    subset.to_json(dest_path, orient="records", lines=True)


if __name__ == "__main__":
    main()
