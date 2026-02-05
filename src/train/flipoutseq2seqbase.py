from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Self

import torch
from safetensors.torch import load_file
from transformers import PreTrainedModel
from transformers.modeling_outputs import Seq2SeqLMOutput

from src.train.linearflipoutadapter import LinearFlipoutAdapter


@dataclass
class Seq2SeqFlipoutLMOutput(Seq2SeqLMOutput):
    """Add KL divergence to Flipout output."""

    kl: torch.FloatTensor | None = None


class FlipoutSeq2SeqBase(PreTrainedModel, ABC):
    """Implements BART with a Bayesian output layer using flipout."""

    @classmethod
    def from_base_pretrained(cls, pretrained_model_name_or_path: str, rho: float = -3) -> Self:
        """Load a base model by using pretrained output weights as prior and posterior."""
        model = super().from_pretrained(pretrained_model_name_or_path)
        model.lm_head = LinearFlipoutAdapter(model.lm_head, rho=rho)
        FlipoutSeq2SeqBase.enable_flipout_last_n_decoder_ffn(model, n=4, rho=rho)

        return model

    @staticmethod
    def enable_flipout_last_n_decoder_ffn(model, n: int, rho: float) -> None:
        for blk in model.decoder.block[-n:]:
            dense = blk.layer[2].DenseReluDense
            dense.wi = LinearFlipoutAdapter(dense.wi, rho=rho)
            dense.wo = LinearFlipoutAdapter(dense.wo, rho=rho)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str) -> Self:
        """Load a previously trained Flipout model."""
        model = super().from_pretrained(pretrained_model_name_or_path)      # will give a warning
                                                                            # mu and rho are not loaded immediately
        model.lm_head = LinearFlipoutAdapter(model.lm_head)

        state_dict_path = Path(pretrained_model_name_or_path) / "model.safetensors"
        state_dict = load_file(state_dict_path)

        with torch.no_grad():                                               # load manually
            model.lm_head.flip.mu_weight.copy_(state_dict["lm_head.flip.mu_weight"])
            model.lm_head.flip.rho_weight.copy_(state_dict["lm_head.flip.rho_weight"])

            for layer in range(20, 24):
                dense = model.decoder.block[layer].layer[2].DenseReluDense
                dense.wi = LinearFlipoutAdapter(dense.wi)
                dense.wo = LinearFlipoutAdapter(dense.wo)

                dense.wi.flip.mu_weight.copy_(state_dict[f"decoder.block.{layer}.layer.2.DenseReluDense.wi.flip.mu_weight"])
                dense.wi.flip.rho_weight.copy_(state_dict[f"decoder.block.{layer}.layer.2.DenseReluDense.wi.flip.rho_weight"])
                dense.wo.flip.mu_weight.copy_(state_dict[f"decoder.block.{layer}.layer.2.DenseReluDense.wo.flip.mu_weight"])
                dense.wo.flip.rho_weight.copy_(state_dict[f"decoder.block.{layer}.layer.2.DenseReluDense.wo.flip.rho_weight"])

        return model

    @property
    @abstractmethod
    def encoder_decoder_params(self) -> list:
        """Get parameters for encoder and decoder (excluding lm_head)."""

    @abstractmethod
    def forward(self, *args, **kwargs) -> tuple | Seq2SeqFlipoutLMOutput:
        """Forward pass - must be implemented by subclasses."""
