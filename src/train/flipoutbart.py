from pathlib import Path
from typing import Self, override

import torch
from bayesian_torch.layers.flipout_layers import LinearFlipout
from safetensors.torch import load_file
from torch.nn import CrossEntropyLoss
from transformers import BartConfig, BartForConditionalGeneration
from transformers.modeling_outputs import Seq2SeqLMOutput
from transformers.models.bart.modeling_bart import shift_tokens_right


class FlipoutBart(BartForConditionalGeneration):
    """Implements BART with a Bayesian output layer using flipout."""

    def __init__(self, config: BartConfig) -> None:
        """Create a Flipout BART model."""
        super().__init__(config)           #  LinearFlipout in ctor would make us lose pretrained weights.
        self.train_size: int | None = getattr(config, "train_size", None)           # for scaling KL

    @classmethod
    def from_bart_pretrained(cls, pretrained_model_name_or_path: str, train_size: int, rho: float = -3) -> Self:
        """Load a BART model by using pretrained output weights as posterior."""
        model = super().from_pretrained(pretrained_model_name_or_path)
        model.train_size = train_size                           # not present in base model
        model.config.train_size = train_size                    # save for the future

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

    @override
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        decoder_input_ids: torch.LongTensor | None = None,
        decoder_attention_mask: torch.LongTensor | None = None,
        head_mask: torch.Tensor | None = None,
        decoder_head_mask: torch.Tensor | None = None,
        cross_attn_head_mask: torch.Tensor | None = None,
        encoder_outputs: list[torch.FloatTensor] | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        decoder_inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
    ) -> tuple | Seq2SeqLMOutput:
        """Perform a forward pass with the Flipout BART model, based on base class implementation."""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            if use_cache:
                print("The `use_cache` argument is changed to `False` since `labels` is provided.")
            use_cache = False
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id,
                )

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        lm_logits, kl = self.lm_head(outputs[0])                                # added unpacking
        lm_logits = lm_logits + self.final_logits_bias.to(lm_logits.device)

        masked_lm_loss = None
        if labels is not None:
            labels = labels.to(lm_logits.device)
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))

            masked_lm_loss += kl / self.train_size                        # ELBO = CE + 1/N KL

        if not return_dict:
            output = (lm_logits, *outputs[1:])
            return ((masked_lm_loss, *output)) if masked_lm_loss is not None else output

        return Seq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )
