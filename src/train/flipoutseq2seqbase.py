from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Self

import torch
from bayesian_torch.layers.flipout_layers import LinearFlipout
from safetensors.torch import load_file
from transformers import PreTrainedModel
from transformers.modeling_outputs import Seq2SeqLMOutput


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

        pretrained_w = model.lm_head.weight.data.clone()        # from pretraining

                                                    # merely replacing output head, pretrained model also no bias
        model.lm_head = LinearFlipout(
            in_features=model.config.d_model, out_features=model.config.vocab_size, posterior_rho_init=rho, bias=False)

                                                    # Set weights AFTER LinearFlipout is fully constructed
        with torch.no_grad():                       # https://docs.pytorch.org/docs/stable/nn.init.html
            model.lm_head.prior_weight_mu.copy_(pretrained_w)
            model.lm_head.mu_weight.copy_(pretrained_w)         # warm start posterior
        return model

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str) -> Self:
        """Load a previously trained Flipout model."""
        model = super().from_pretrained(pretrained_model_name_or_path)      # will give a warning
                                                                            # mu and rho are not loaded immediately
        model.lm_head = LinearFlipout(model.config.d_model, model.config.vocab_size, bias=False)
        state_dict_path = Path(pretrained_model_name_or_path) / "model.safetensors"
        state_dict = load_file(state_dict_path)

        with torch.no_grad():                                               # load manually
            model.lm_head.mu_weight.copy_(state_dict["lm_head.mu_weight"])
            model.lm_head.rho_weight.copy_(state_dict["lm_head.rho_weight"])

        return model

    @property
    @abstractmethod
    def encoder_decoder_params(self) -> list:
        """Get parameters for encoder and decoder (excluding lm_head)."""

    @abstractmethod
    def forward(self, *args, **kwargs) -> tuple | Seq2SeqFlipoutLMOutput:
        """Forward pass - must be implemented by subclasses."""
