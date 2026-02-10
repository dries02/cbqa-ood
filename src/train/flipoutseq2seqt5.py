from pathlib import Path
from typing import Self, override

import torch
from safetensors.torch import load_file
from transformers import T5ForConditionalGeneration

from src.train.flipoutseq2seqbase import FlipoutSeq2SeqBase
from src.train.linearflipoutadapter import LinearFlipoutAdapter


class FlipoutSeq2SeqT5(FlipoutSeq2SeqBase, T5ForConditionalGeneration):
    """Implements T5 with LinearFlipout layers."""

    @override
    @classmethod
    def from_base_pretrained(cls, pretrained_model_name_or_path: str, rho: float = -3.0) -> Self:
        """Load a base model by using pretrained output weights as prior and posterior."""
        model = super().from_pretrained(pretrained_model_name_or_path)

        model.lm_head = LinearFlipoutAdapter(model.lm_head, rho=rho)

        cls._enable_flipout_last_n_encoder(model, n=3, rho=rho)
        cls._enable_flipout_last_n_decoder(model, n=3, rho=rho)

        return model

    @staticmethod
    def _enable_flipout_last_n_encoder(model: "FlipoutSeq2SeqT5", n: int, rho: float) -> None:
        """Enable Flipout in the last `n` encoder blocks."""
        for blk in model.encoder.block[-n:]:
            self_attention = blk.layer[0].SelfAttention
            self_attention.q = LinearFlipoutAdapter(self_attention.q, rho=rho)
            self_attention.k = LinearFlipoutAdapter(self_attention.k, rho=rho)
            self_attention.v = LinearFlipoutAdapter(self_attention.v, rho=rho)
            self_attention.o = LinearFlipoutAdapter(self_attention.o, rho=rho)

            dense = blk.layer[1].DenseReluDense
            dense.wi = LinearFlipoutAdapter(dense.wi, rho=rho)
            dense.wo = LinearFlipoutAdapter(dense.wo, rho=rho)

    @staticmethod
    def _enable_flipout_last_n_decoder(model: "FlipoutSeq2SeqT5", n: int, rho: float) -> None:
        """Enable Flipout in the last `n` decoder blocks."""
        for blk in model.decoder.block[-n:]:
            self_attention = blk.layer[0].SelfAttention
            self_attention.q = LinearFlipoutAdapter(self_attention.q, rho=rho)
            self_attention.k = LinearFlipoutAdapter(self_attention.k, rho=rho)
            self_attention.v = LinearFlipoutAdapter(self_attention.v, rho=rho)
            self_attention.o = LinearFlipoutAdapter(self_attention.o, rho=rho)

            enc_dec_attention = blk.layer[1].EncDecAttention
            enc_dec_attention.q = LinearFlipoutAdapter(enc_dec_attention.q, rho=rho)
            enc_dec_attention.k = LinearFlipoutAdapter(enc_dec_attention.k, rho=rho)
            enc_dec_attention.v = LinearFlipoutAdapter(enc_dec_attention.v, rho=rho)
            enc_dec_attention.o = LinearFlipoutAdapter(enc_dec_attention.o, rho=rho)

            dense = blk.layer[2].DenseReluDense
            dense.wi = LinearFlipoutAdapter(dense.wi, rho=rho)
            dense.wo = LinearFlipoutAdapter(dense.wo, rho=rho)

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
