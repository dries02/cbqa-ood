import random
from dataclasses import dataclass, field

import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase
from transformers.modeling_outputs import Seq2SeqLMOutput


@dataclass
class _TrieNode:
    """Store all answers by prefix in a Trie."""

    children: dict[int, "_TrieNode"] = field(default_factory=dict)

    def add_answer(self, answer: list[int], eos_token_id: int) -> None:
        """Add answer to the root."""
        curr = self                                     # points to the leaf
        for token_id in answer:
            if token_id not in curr.children:           # new continuation
                curr.children[token_id] = _TrieNode()

            if token_id == eos_token_id:                # ignore padding
                break
            curr = curr.children[token_id]              # move to child

    def traverse(self, reference_answer: list[int], eos_token_id: int) -> list[int]:
        """Create soft labels by traversing the trie with a reference answer."""
        curr = self
        soft_labels = []
        for token_id in reference_answer:
            soft_labels.append(list(curr.children.keys()))
            if token_id == eos_token_id:
                return soft_labels

            curr = curr.children[token_id]

        msg = "This should never be reached"
        raise ValueError(msg)


class QADatasetTrainSoft(Dataset):
    """QA Dataset with pre-tokenized inputs and labels for training using soft labels."""

    def __init__(self, train_df: pd.DataFrame, tokenizer: PreTrainedTokenizerBase, *,
                 remove_bos: bool, prefix: str, max_len_q: int = 32, max_len_a: int = 32, n_answers: int = 5) -> None:
        """Pre-tokenize everything at construction."""
        super().__init__()
        questions = train_df["question"].apply(lambda q: prefix + q).tolist()
        self.encodings_q = tokenizer(questions, max_length=max_len_q,
                                     truncation=True, padding="max_length", return_tensors="pt")
        self.hard_labels, self.soft_labels = self._tokenize_labels(
            train_df["answers"], tokenizer, max_len_a, n_answers, remove_bos=remove_bos)

    @staticmethod
    def _tokenize_labels(all_answers: pd.Series, tokenizer: PreTrainedTokenizerBase, max_len_a: int, n_answers: int,
                         *, remove_bos: bool) -> tuple[list[torch.Tensor], list[list[list[list[int]]]]]:
                    #               ^question           ^question ^answer ^position ^valid_tokens
        hard_labels = []
        soft_labels = []
        for answers in all_answers:
            tokens = tokenizer(answers[:n_answers], max_length=max_len_a, truncation=True,
                padding="max_length", return_tensors="pt",
            ).input_ids

            if remove_bos:
                tokens = tokens[:, 1:]

            root = _TrieNode()
            for ans in tokens.tolist():
                root.add_answer(ans, tokenizer.eos_token_id)    # identify all prefixes

            answer_soft_labels = [root.traverse(ans, tokenizer.eos_token_id) for ans in tokens.tolist()]
            tokens[tokens == tokenizer.pad_token_id] = -100     # Mask padding, does not affect soft labels
            hard_labels.append(tokens)
            soft_labels.append(answer_soft_labels)

        return hard_labels, soft_labels

    def __len__(self) -> int:
        """Count number of samples in dataset."""
        return len(self.hard_labels)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | list[list[int]]]:
        """Fetch sample from dataset."""
        ans_idx = random.randrange(0, len(self.hard_labels[idx]))     # sample an answer
        return {
            "input_ids":      self.encodings_q.input_ids[idx],
            "attention_mask": self.encodings_q.attention_mask[idx],
            "labels":         self.hard_labels[idx][ans_idx],
            "soft_labels":    self.soft_labels[idx][ans_idx],
        }

        # input is a list of dicts of size "batch_size", where each dict comes from __getitem__.
    @staticmethod
    def collate_fn(batch: list[dict[str, torch.Tensor | list[list[int]]]]) -> dict[str, torch.Tensor | list]:
        """Align the lists for dataloader. Train requires special care since sparse labels are not stackable."""
        return {
            "input_ids":      torch.stack([ex["input_ids"]      for ex in batch], 0),
            "attention_mask": torch.stack([ex["attention_mask"] for ex in batch], 0),
            "labels":         torch.stack([ex["labels"]         for ex in batch], 0),
            "soft_labels":    [ex["soft_labels"]                for ex in batch],
        }


def compute_kl_soft_loss(outputs: Seq2SeqLMOutput, batch: dict, batch_gpu: dict) -> torch.Tensor:
    """Compute KL divergence loss with soft labels."""
    log_probs = torch.log_softmax(outputs.logits, dim=-1)   # shape: (batch_size, max_ans_len, vocab_size)
    soft_targets = torch.zeros_like(log_probs, device=log_probs.device)

    soft_labels = batch["soft_labels"]

    for b in range(log_probs.shape[0]):                     # batch_size
        for t in range(len(soft_labels[b])):
            allowed = soft_labels[b][t]
            prob = 1 / len(allowed)
            soft_targets[b, t, allowed] = prob

    mask = batch_gpu["labels"] != -100                      # ignore index
    log_probs_flat = log_probs[mask]                        # (N, vocab_size)
    soft_targets_flat = soft_targets[mask]                  # (N, vocab_size)
    kl_loss_fn = nn.KLDivLoss(reduction="batchmean")
    return kl_loss_fn(log_probs_flat, soft_targets_flat)
