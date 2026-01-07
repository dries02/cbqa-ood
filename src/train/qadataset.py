import random

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase


class QADatasetTrain(Dataset):
    """QA Dataset with pre-tokenized inputs and labels for training.

    Training requires labels to be pretokenized as well.
    Since multiple answers may be correct, labels can be considered stochastic.
    """

    def __init__(self, df: pd.DataFrame, tokenizer: PreTrainedTokenizerBase, *,
                 use_stochastic_labels: bool = False, max_len_q: int = 32, max_len_a: int = 32) -> None:
        """Pre-tokenize everything at construction."""
        super().__init__()
        self.use_stochastic_labels = use_stochastic_labels

        self.encodings_q = tokenizer(
            df["question"].to_list(), max_length=max_len_q, truncation=True, padding="max_length", return_tensors="pt")

        def tokenize_labels(answers: list[str], tokenizer: PreTrainedTokenizerBase, max_len: int) -> torch.Tensor:
            tokens = tokenizer(
                answers, max_length=max_len, truncation=True,
                padding="max_length", return_tensors="pt",
            ).input_ids[:, 1:]                                  # Remove <bos>

            tokens[tokens == tokenizer.pad_token_id] = -100     # Mask padding
            return tokens

        self.labels = [
            tokenize_labels(answers, tokenizer, max_len_a)
            for answers in df["answers"]
        ]

    def __len__(self) -> int:
        """Count number of samples in dataset."""
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Fetch sample from dataset."""
        label = random.choice(self.labels[idx]) if self.use_stochastic_labels else self.labels[idx][0]
        return {
            "input_ids": self.encodings_q.input_ids[idx],
            "attention_mask": self.encodings_q.attention_mask[idx],
            "labels": label,
        }


class QADatasetEval(Dataset):
    """QA Dataset with pre-tokenized inputs and labels for evaluation."""

    def __init__(self, df: pd.DataFrame, tokenizer: PreTrainedTokenizerBase, *, max_len_q: int = 32) -> None:
        """Pre-tokenize everything at initialization."""
        super().__init__()
        self.encodings_q = tokenizer(
            df["question"].to_list(), max_length=max_len_q, truncation=True, padding="max_length", return_tensors="pt")
        self.labels = df["answers"]

    def __len__(self) -> int:
        """Count number of samples in dataset."""
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | list[str]]:
        """Fetch sample from dataset."""
        return {
            "input_ids": self.encodings_q["input_ids"][idx],
            "attention_mask": self.encodings_q["attention_mask"][idx],
            "labels": self.labels.iloc[idx],                            # safe indexing
        }

        # input is a list of dicts of size "batch_size", where each dict comes from __getitem__.
    @staticmethod
    def collate_fn(batch: list[dict[str, torch.Tensor | list[str]]]) -> dict[str, torch.Tensor | list[list[str]]]:
        """Align the lists of strings for data loader. Eval requires special care since labels are not stackable."""
        return {
            "input_ids":      torch.stack([ex["input_ids"]      for ex in batch], 0),
            "attention_mask": torch.stack([ex["attention_mask"] for ex in batch], 0),
            "labels":         [ex["labels"]                     for ex in batch],
        }
