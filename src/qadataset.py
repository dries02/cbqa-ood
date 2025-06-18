import pandas as pd
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase


class QADataset(Dataset):
    """Implements a Dataset for closed-book question answering.

    Tokenization occurs on the fly for ease of implementation. Takes less memory,
    in terms of time should be negligible.
    Multiple answers is not supported (for now). Could be implemented by passing an argument
    to __init__, then at training time return random answer, at inference time return all
    possible answers
    """

    def __init__(self, df: pd.DataFrame, tokenizer: PreTrainedTokenizerBase) -> None:
        """Create a QA dataset."""
        super().__init__()
        self.df = df
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        """Return number of samples in dataset."""
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        """Fetch sample from dataset."""
        sample = self.df.iloc[idx]
        tok_q = self.tokenizer(
            sample["question"],
            max_length=64,                  # should be enough
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        tok_a = self.tokenizer(
            sample["answers"][0],           # multi answer is not supported...
            max_length=32,                  # maybe increase this
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        return {
            "input_ids": tok_q["input_ids"].squeeze(0),
            "attention_mask": tok_q["attention_mask"].squeeze(0),
            "labels": tok_a["input_ids"].squeeze(0),
        }
