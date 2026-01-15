from typing import override

import torch
from torch.nn import CrossEntropyLoss
from transformers import BartForConditionalGeneration
from transformers.models.bart.modeling_bart import shift_tokens_right

from src.train.flipoutseq2seqbase import FlipoutSeq2SeqBase, Seq2SeqFlipoutLMOutput


class FlipoutSeq2SeqBart(FlipoutSeq2SeqBase, BartForConditionalGeneration):
    """Implements BART with a Bayesian output layer using flipout."""

    @property
    def encoder_decoder_params(self) -> list:
        return self.model.parameters()

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
    ) -> tuple | Seq2SeqFlipoutLMOutput:
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

        if not return_dict:
            output = (lm_logits, *outputs[1:])
            return ((masked_lm_loss, *output)) if masked_lm_loss is not None else output

        return Seq2SeqFlipoutLMOutput(
            loss=masked_lm_loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
            kl=kl,                                                              # added KL divergence
        )
