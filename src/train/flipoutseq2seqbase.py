from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Self

import torch
from transformers import PreTrainedModel
from transformers.modeling_outputs import Seq2SeqLMOutput


@dataclass
class Seq2SeqFlipoutLMOutput(Seq2SeqLMOutput):
    """Add KL divergence to Flipout LM output."""

    kl: torch.FloatTensor | None = None


class FlipoutSeq2SeqBase(PreTrainedModel, ABC):
    """Implements BART with a Bayesian output layer using flipout."""

    @classmethod
    @abstractmethod
    def from_base_pretrained(cls, *args, **kwargs) -> Self:
        """Load a base model by using pretrained output weights as prior and posterior."""

    @abstractmethod
    def fetch_kl(self) -> torch.Tensor:
        """Fetch KL terms - must be implemented by child classes."""

    def forward(self, *args, **kwargs) -> Seq2SeqFlipoutLMOutput:
        """Forward pass. Unsafe if `return_dict` is passed."""
        base_out = super().forward(*args, **kwargs)
        return Seq2SeqFlipoutLMOutput(**base_out, kl=self.fetch_kl())
