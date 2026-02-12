from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Self

import torch
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import Seq2SeqLMOutput

from src.train.linearflipoutadapter import LinearFlipoutAdapter


@dataclass
class Seq2SeqFlipoutLMOutput(Seq2SeqLMOutput):
    """Add KL divergence to Flipout LM output."""

    kl: torch.FloatTensor | None = None


class FlipoutSeq2SeqBase(PreTrainedModel, ABC):
    """Implements a Seq2Seq LM with LinearFlipout layers."""

    @classmethod
    @abstractmethod
    def from_base_pretrained(
        cls, pretrained_model_name_or_path: str, config: PretrainedConfig, rho: float) -> Self:
        """Load a base model by using pretrained output weights as prior and posterior."""

    def _fetch_kl(self) -> torch.Tensor:
            # slightly inefficient but loose coupling to model architecture
        kls = [layer.kl_loss() for layer in self.modules() if isinstance(layer, LinearFlipoutAdapter)]
        if not kls:
            msg = "No Flipout layers found."
            raise ValueError(msg)

        return torch.stack(kls).mean()

    def forward(self, *args, **kwargs) -> Seq2SeqFlipoutLMOutput:
        """Forward pass. Unsafe if `return_dict` is passed."""
        base_out = super().forward(*args, **kwargs)
        return Seq2SeqFlipoutLMOutput(**base_out, kl=self._fetch_kl())
