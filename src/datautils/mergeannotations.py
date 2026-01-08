"""Inspect the annotated test data from Lewis et al."""

import ast
import json
from argparse import ArgumentParser, Namespace
from enum import Enum
from pathlib import Path

import pandas as pd


class LabelType(Enum):
    """Used to index the list below."""

    ID = 1                      # Question memorization
    FAR_OOD = 4                 # QA generalization
    NEAR_OOD = 5                # Answer classification

ANNOTATIONS = [
    "total",                    # 0 present in every test sample, referred in paper as "Total"
    "question_overlap",         # 1 for Question memorization, referred in paper "Question Overlap"
    "no_question_overlap",      # 2 ...
    "answer_overlap",           # 3 ...
    "no_answer_overlap",        # 4 for QA generalization, referred in paper as "No Overlap"
    "answer_overlap_only",      # 5 for Answer classification, referred in paper as "Answer Overlap Only"
                                #   so Question-overlap pairs are excluded
]


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=["nq", "webquestions", "triviaqa"], required=True)
    return parser.parse_args()


def contains_unique_label(annotations: list[str]) -> bool:
    """Check if annotations contain exactly one (useful) label."""
    return ((ANNOTATIONS[LabelType.ID.value] in annotations)
            + (ANNOTATIONS[LabelType.NEAR_OOD.value] in annotations)
            + (ANNOTATIONS[LabelType.FAR_OOD.value] in annotations)) == 1


def prettify_labels(annotations: list[str]) -> str:
    """Convert the annotations to useful labels for OOD detection."""
    if not contains_unique_label(annotations):
        return "other"
    if ANNOTATIONS[LabelType.ID.value] in annotations:
        return "in"
    if ANNOTATIONS[LabelType.NEAR_OOD.value] in annotations:
        return "near-ood"
    if ANNOTATIONS[LabelType.FAR_OOD.value] in annotations:
        return "far-ood"

    msg = f"annotations contains no label: {annotations}"
    raise ValueError(msg)


def load_labels(annotations_path: Path) -> pd.DataFrame:
    """Load the annotated labels from .jsonl file, excluding the ids/row numbers.

    Returns:
        pd.DataFrame: The loaded annotated labels.
    """
    with Path.open(annotations_path) as file:
        return pd.DataFrame(json.loads(line) for line in file)["labels"]


def main() -> None:
    """Program to load annotated test set (ID/near-OOD/far-OOD samples) from a given data set."""
    args = parse_args()
    base_path = Path("data") / args.dataset

    test_df = pd.read_csv(base_path / f"{args.dataset}-test.qa.csv", sep="\t", names=["question", "answers"])
                                    # convert string repr of a list to a list...
    test_df["answers"] = test_df["answers"].apply(ast.literal_eval)

                                    # we are very sure that this aligns well
    test_df["labels"] = load_labels(base_path / f"{args.dataset}-annotations.jsonl")
    test_df["labels"] = test_df["labels"].apply(prettify_labels)     # human-readable
    test_df.to_json(base_path / f"{args.dataset}-test.jsonl", orient="records", lines=True)

if __name__ == "__main__":
    main()
