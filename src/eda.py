"""Inspect the annotated test data from Lewis et al."""

import argparse
import json
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

OOD_MAP = {             # for indexing parser output
    "far": LabelType.FAR_OOD,
    "near": LabelType.NEAR_OOD,
}


def parse_args() -> argparse.Namespace:
    """Create a parser.

    Returns:
        argparse.Namespace: A parser that expects the user to enter a data set identifier and OOD type.
    """
    parser = argparse.ArgumentParser(description="Process QA data set selection and OOD choice.")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["nq", "webquestions", "triviaqa"],
        required=True,
        help=(
            "Specify which dataset to use: nq (Natural Questions), "
            "webquestions (WebQuestions), or triviaqa (TriviaQA)."
        ),
    )
    parser.add_argument(
        "--ood_type",
        type=str,
        choices=["far", "near"],
        required=True,
        help="Specify the type of OOD: far (QA generalization) or near (answer classification).",
    )

    return parser.parse_args()


def load_labels(base_path: str, annotations_extension: str) -> pd.DataFrame:
    """Load the annotated labels from .jsonl file, excluding the ids/row numbers.

    Returns:
        pd.DataFrame: The loaded annotated labels.
    """
    with Path.open(base_path + annotations_extension) as file:
        return pd.DataFrame(json.loads(line) for line in file)["labels"]


def encode_labels(test_df: pd.DataFrame, ood_type: LabelType) -> pd.DataFrame:
    """Encode labels into ID / OD and keep only those samples.

    Returns:
        pd.DataFrame: df containing binary "labels" column (ID=0, OD=1) with other samples dropped.
    """
    id_str = ANNOTATIONS[LabelType.ID.value]
    od_str = ANNOTATIONS[ood_type.value]
    far_ood_str = ANNOTATIONS[LabelType.FAR_OOD.value]

                                    # check if either of the labels is present
    test_df = test_df[test_df["labels"].apply(lambda x: id_str in x or od_str in x)]

                                    # in fact there is overlap between ID and FAR_OOD somehow.
                                    # keep ID the same independent of OD (near or far)
    overlap_mask = test_df["labels"].apply(lambda x: not (id_str in x and far_ood_str in x))
    print(f"Removing {len(overlap_mask) - overlap_mask.sum()} overlapping samples between ID and far OOD")
    test_df = test_df[overlap_mask]

                                    # encode OD = True, ID = False
    test_df.loc[:, "labels"] = test_df["labels"].apply(lambda x: od_str in x)
    return test_df


def load_test_set(
        base_path: str,
        ood_type: LabelType,
        test_extension: str = "test.qa.csv",
        annotations_extension: str = "annotations.jsonl",
        ) -> pd.DataFrame:
    """Read (q,a) pairs with annotated label from .csv and .jsonl file.

    Returns:
        pd.DataFrame: The loaded test set with encoded ID (0) / OD (1) samples.
    """
    data = pd.read_csv(base_path + test_extension, sep="\t", names=["question", "answers"])
                                    # we are very sure that this aligns well
    data["labels"] = load_labels(base_path, annotations_extension)
    return encode_labels(data, ood_type)


def main() -> None:
    """Program to load annotated test set (ID/OD samples) from a given data set."""
    args = parse_args()
    base_path = f"data/{args.dataset}/{args.dataset}-"

    test_df = load_test_set(base_path, OOD_MAP[args.ood_type])

    n_od = test_df["labels"].sum()
    print(f"{len(test_df) - n_od} ID samples, {n_od} OOD samples")
                                    # lists might look like strings in view but its ok
    test_df.to_parquet(base_path + f"merged-{args.ood_type}.parquet", index=False)

if __name__ == "__main__":
    main()
