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
    """Create a parser.

    Returns:
        argparse.Namespace: A parser that expects the user to enter a data set identifier and OOD type.
    """
    parser = ArgumentParser(description="Process QA data set selection and OOD choice.")
    parser.add_argument("--dataset", type=str, choices=["nq", "webquestions", "triviaqa"], required=True,
        help=(
            "Specify which dataset to use: nq (Natural Questions), webquestions (WebQuestions), or triviaqa (TriviaQA)."
        ),
    )

    return parser.parse_args()


def contains_unique_label(annotations: list[str]) -> bool:
    """Check if annotations contain exactly one (useful) label."""
    return ((ANNOTATIONS[LabelType.ID.value] in annotations)
            + (ANNOTATIONS[LabelType.NEAR_OOD.value] in annotations)
            + (ANNOTATIONS[LabelType.FAR_OOD.value] in annotations)) == 1


def prettify_labels(annotations: list[str]) -> str:
    """Convert the annotations to useful labels for OOD detection."""
    if ANNOTATIONS[LabelType.ID.value] in annotations:
        return "in"
    if ANNOTATIONS[LabelType.NEAR_OOD.value] in annotations:
        return "near-ood"
    if ANNOTATIONS[LabelType.FAR_OOD.value] in annotations:
        return "far-ood"

    msg = f"annotations contains no label: {annotations}"
    raise ValueError(msg)


def load_labels(base_path: str, annotations_extension: str) -> pd.DataFrame:
    """Load the annotated labels from .jsonl file, excluding the ids/row numbers.

    Returns:
        pd.DataFrame: The loaded annotated labels.
    """
    with Path.open(base_path + annotations_extension) as file:
        return pd.DataFrame(json.loads(line) for line in file)["labels"]


def encode_labels(test_df: pd.DataFrame) -> pd.DataFrame:
    """Encode labels into ID / OD and keep only those samples.

    Returns:
        pd.DataFrame: df containing id/{near/far}-ood "labels" column with other samples dropped.
    """
    test_df = test_df[test_df["labels"].apply(contains_unique_label)]       # exactly one label present
    test_df.loc[:, "labels"] = test_df["labels"].apply(prettify_labels)     # human-readable
    return test_df


def load_test_set(
        base_path: str, test_extension: str = "test.qa.csv", annotations_extension: str = "annotations.jsonl") -> pd.DataFrame:
    """Read (q,a) pairs with annotated label from .csv and .jsonl file.

    Returns:
        pd.DataFrame: The loaded test set with encoded ID (0) / OD (1) samples.
    """
    data = pd.read_csv(base_path + test_extension, sep="\t", names=["question", "answers"])
                                    # convert string repr of a list to a list...
    data["answers"] = data["answers"].apply(ast.literal_eval)
                                    # we are very sure that this aligns well
    data["labels"] = load_labels(base_path, annotations_extension)
    return encode_labels(data)


def main() -> None:
    """Program to load annotated test set (ID/near-OOD/far-OOD samples) from a given data set."""
    args = parse_args()
    base_path = f"data/{args.dataset}/{args.dataset}-"

    test_df = load_test_set(base_path)
    test_df.to_json(base_path + "test.jsonl", orient="records", lines=True)

if __name__ == "__main__":
    main()
