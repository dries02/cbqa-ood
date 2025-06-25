import random

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase


class QADataset(Dataset):
    """Implements a Dataset for closed-book question answering.

    Tokenization occurs on the fly for ease of implementation. Takes less memory,
    in terms of time should be negligible.

    Multiple answers is supported in the sense that a random item is returned. This makes the labels
    stochastic. This is more space-efficient than exploding the answers and ensures no imbalance between
    questions. For the validation/test set one might desire deterministic answers.
    """

    def __init__(self, df: pd.DataFrame, tokenizer: PreTrainedTokenizerBase, *, is_train: bool) -> None:
        """Create a QA dataset."""
        super().__init__()
        self.df = df
        self.tokenizer = tokenizer
        self.is_train = is_train

    def __len__(self) -> int:
        """Return number of samples in dataset."""
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | list[str]]:
        """Fetch sample from dataset."""
        sample = self.df.iloc[idx]
        tok_q = self.tokenizer(
            sample["question"],
            max_length=32,                          # should be more than enough, ensure no truncation
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        if not self.is_train:                       # early return all the answers
            return {
                "input_ids": tok_q["input_ids"].squeeze(0),
                "attention_mask": tok_q["attention_mask"].squeeze(0),
                "labels": sample["answers"],
            }

        tok_a = self.tokenizer(
            # sample["answers"][0],       # support multi-answer by sampling
            random.choice(sample["answers"]),       # support multi-answer by sampling
            max_length=32,                          # should be enough, answers are typically very very short
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        return {
            "input_ids": tok_q["input_ids"].squeeze(0),
            "attention_mask": tok_q["attention_mask"].squeeze(0),
            "labels": tok_a["input_ids"].squeeze(0),
        }


def eval_collate_fn(batch: list[dict]) -> dict:
    """Align the lists of strings for data loader."""
    input_ids      = torch.stack([ex["input_ids"]      for ex in batch], 0)
    attention_mask = torch.stack([ex["attention_mask"] for ex in batch], 0)
    labels         = [ex["labels"] for ex in batch]   # list of list[str]
    return {
        "input_ids":      input_ids,
        "attention_mask": attention_mask,
        "labels":         labels,
    }
