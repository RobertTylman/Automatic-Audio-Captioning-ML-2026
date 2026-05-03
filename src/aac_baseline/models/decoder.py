from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.bart.modeling_bart import BartConfig, BartDecoder


@dataclass
class DecoderOutput:
    loss: torch.Tensor | None
    logits: torch.Tensor


def shift_tokens_right(
    labels: torch.Tensor,
    pad_token_id: int,
    decoder_start_token_id: int,
) -> torch.Tensor:
    shifted = labels.new_zeros(labels.shape)
    shifted[:, 0] = decoder_start_token_id
    shifted[:, 1:] = labels[:, :-1]
    shifted.masked_fill_(shifted == -100, pad_token_id)
    return shifted


class BartCaptionDecoder(nn.Module):
    """
    We use a BART-style decoder with cross-attention over audio features.
    """

    def __init__(self, config: dict) -> None:
        super().__init__()
        self.config = BartConfig(
            vocab_size=config["vocab_size"],
            d_model=config["d_model"],
            decoder_layers=config["decoder_layers"],
            decoder_ffn_dim=config["decoder_ffn_dim"],
            decoder_attention_heads=config["decoder_attention_heads"],
            max_position_embeddings=config["max_position_embeddings"],
            dropout=config["dropout"],
            attention_dropout=config["attention_dropout"],
            activation_dropout=config["activation_dropout"],
            pad_token_id=config["pad_token_id"],
            bos_token_id=config["bos_token_id"],
            eos_token_id=config["eos_token_id"],
            decoder_start_token_id=config["decoder_start_token_id"],
        )

        self.token_embedding = nn.Embedding(
            self.config.vocab_size,
            self.config.d_model,
            padding_idx=self.config.pad_token_id,
        )
        self.decoder = BartDecoder(self.config)
        self.decoder.embed_tokens = self.token_embedding
        self.lm_head = nn.Linear(self.config.d_model, self.config.vocab_size, bias=False)
        self.final_logits_bias = nn.Parameter(torch.zeros(1, self.config.vocab_size))

    def forward(
        self,
        encoder_hidden_states: torch.Tensor,
        encoder_padding_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> DecoderOutput:
        # Teacher forcing: the decoder receives the target sentence shifted to
        # the right, and learns to predict the next token at each time step.
        decoder_input_ids = shift_tokens_right(
            labels=labels,
            pad_token_id=self.config.pad_token_id,
            decoder_start_token_id=self.config.decoder_start_token_id,
        )

        decoder_attention_mask = decoder_input_ids.ne(self.config.pad_token_id).long()
        encoder_attention_mask = (~encoder_padding_mask).long()

        outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            return_dict=True,
        )

        logits = self.lm_head(outputs.last_hidden_state) + self.final_logits_bias
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=-100,
        )
        return DecoderOutput(loss=loss, logits=logits)

    @torch.no_grad()
    def greedy_decode(
        self,
        encoder_hidden_states: torch.Tensor,
        encoder_padding_mask: torch.Tensor,
        max_length: int = 32,
    ) -> torch.Tensor:
        batch_size = encoder_hidden_states.size(0)
        generated = torch.full(
            (batch_size, 1),
            fill_value=self.config.decoder_start_token_id,
            dtype=torch.long,
            device=encoder_hidden_states.device,
        )

        encoder_attention_mask = (~encoder_padding_mask).long()

        for _ in range(max_length - 1):
            decoder_attention_mask = generated.ne(self.config.pad_token_id).long()
            outputs = self.decoder(
                input_ids=generated,
                attention_mask=decoder_attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                return_dict=True,
            )
            next_token = self.lm_head(outputs.last_hidden_state[:, -1]) + self.final_logits_bias
            next_token = next_token.argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)

            if torch.all(next_token.squeeze(1) == self.config.eos_token_id):
                break

        return generated
