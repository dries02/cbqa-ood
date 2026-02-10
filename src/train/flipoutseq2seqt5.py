from pathlib import Path
from typing import Self, override

import torch
from safetensors.torch import load_file
from torch import nn
from transformers import PretrainedConfig, T5ForConditionalGeneration

from src.train.flipoutseq2seqbase import FlipoutSeq2SeqBase
from src.train.linearflipoutadapter import LinearFlipoutAdapter


class FlipoutSeq2SeqT5(FlipoutSeq2SeqBase, T5ForConditionalGeneration):
    """Implements T5 with LinearFlipout layers."""

    @classmethod
    def _replace_flipout(cls, parent: nn.Module, rho: float) -> None:
        """Recursively replace all Linear layers with LinearFlipout layers."""
        for name, child in parent.named_children():
            if isinstance(child, nn.Linear):
                setattr(parent, name, LinearFlipoutAdapter(child, rho))
            else:
                cls._replace_flipout(child, rho)

    @override
    @classmethod
    def from_base_pretrained(
        cls, pretrained_model_name_or_path: str, config: PretrainedConfig, rho: float = -3.0,
    ) -> Self:
        """Load a base model by using pretrained output weights as prior and posterior."""
        model = super().from_pretrained(pretrained_model_name_or_path, config=config)
        cls._replace_flipout(model, rho)

        return model

    @override
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
