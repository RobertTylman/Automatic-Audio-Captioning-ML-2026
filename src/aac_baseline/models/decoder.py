from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BartForConditionalGeneration
from transformers.models.bart.modeling_bart import BartConfig, BartDecoder


@dataclass
class DecoderOutput:
    loss: torch.Tensor | None
    logits: torch.Tensor


class AudioConditioningProjection(nn.Module):
    """
    Optional adapter between the audio stack and BART cross-attention.

    Keeping this disabled preserves the previous behavior. Enabling a small- or
    zero-initialized projection lets the pretrained BART decoder receive audio
    conditioning gradually instead of seeing full-strength non-text embeddings
    from step one.
    """

    def __init__(self, d_model: int, config: dict | None) -> None:
        super().__init__()
        config = config or {}
        projection_type = config.get("type", "identity")
        self.projection_type = projection_type

        if projection_type == "identity":
            self.norm = nn.Identity()
            self.proj = nn.Identity()
            return

        if projection_type != "linear":
            raise ValueError(
                "Unsupported audio_projection type. "
                f"Expected 'identity' or 'linear', got {projection_type!r}."
            )

        self.norm = nn.LayerNorm(d_model) if config.get("layer_norm", False) else nn.Identity()
        self.proj = nn.Linear(d_model, d_model, bias=config.get("bias", False))
        self._initialize_projection(
            init=config.get("init", "default"),
            init_std=config.get("init_std", 1e-3),
        )

    def _initialize_projection(self, init: str, init_std: float) -> None:
        if init == "default":
            return

        if init == "zero":
            nn.init.zeros_(self.proj.weight)
        elif init == "small_normal":
            nn.init.normal_(self.proj.weight, mean=0.0, std=init_std)
        else:
            raise ValueError(
                "Unsupported audio_projection init. "
                f"Expected 'default', 'zero', or 'small_normal', got {init!r}."
            )

        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)

    def forward(self, encoder_hidden_states: torch.Tensor) -> torch.Tensor:
        return self.proj(self.norm(encoder_hidden_states))


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
        self.label_smoothing = config.get("label_smoothing", 0.0)
        self.audio_projection = AudioConditioningProjection(
            d_model=config["d_model"],
            config=config.get("audio_projection"),
        )
        pretrained_name = config.get("pretrained_model_name_or_path")
        if pretrained_name:
            pretrained = BartForConditionalGeneration.from_pretrained(pretrained_name)
            self.config = pretrained.config
            self.config.pad_token_id = config["pad_token_id"]
            self.config.bos_token_id = config["bos_token_id"]
            self.config.eos_token_id = config["eos_token_id"]
            self.config.decoder_start_token_id = config["decoder_start_token_id"]

            if self.config.d_model != config["d_model"]:
                raise ValueError(
                    "Pretrained BART d_model does not match the audio stack. "
                    f"Got pretrained d_model={self.config.d_model}, "
                    f"configured d_model={config['d_model']}."
                )

            self.decoder = pretrained.model.decoder
            self.lm_head = pretrained.lm_head
            self.final_logits_bias = nn.Parameter(
                pretrained.final_logits_bias.clone().detach()
            )
            self._freeze_pretrained_modules(config.get("freeze_pretrained_modules", []))
        else:
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

    @staticmethod
    def _set_trainable(module: nn.Module, trainable: bool) -> None:
        for parameter in module.parameters():
            parameter.requires_grad = trainable

    def _normalize_freeze_modules(self, freeze_modules) -> set[str]:
        if freeze_modules is None:
            return set()
        if isinstance(freeze_modules, dict):
            return {
                str(module_name)
                for module_name, should_freeze in freeze_modules.items()
                if should_freeze
            }
        if isinstance(freeze_modules, str):
            return {freeze_modules}
        return {str(module_name) for module_name in freeze_modules}

    def _freeze_pretrained_modules(self, freeze_modules) -> None:
        modules = self._normalize_freeze_modules(freeze_modules)
        if not modules:
            return

        aliases = {
            "self_attention": "self_attn",
            "feed_forward": "ffn",
            "feedforward": "ffn",
            "cross_attention": "encoder_attn",
            "cross_attn": "encoder_attn",
            "output_projection": "lm_head",
            "output_head": "lm_head",
            "all_decoder": "decoder",
        }
        modules = {aliases.get(module_name, module_name) for module_name in modules}

        if "all" in modules or "decoder" in modules:
            self._set_trainable(self.decoder, trainable=False)

        if "embeddings" in modules:
            self._set_trainable(self.decoder.embed_tokens, trainable=False)
            self._set_trainable(self.decoder.embed_positions, trainable=False)
            if hasattr(self.decoder, "layernorm_embedding"):
                self._set_trainable(self.decoder.layernorm_embedding, trainable=False)

        for layer in self.decoder.layers:
            if "self_attn" in modules:
                self._set_trainable(layer.self_attn, trainable=False)
            if "encoder_attn" in modules:
                self._set_trainable(layer.encoder_attn, trainable=False)
            if "ffn" in modules:
                self._set_trainable(layer.fc1, trainable=False)
                self._set_trainable(layer.fc2, trainable=False)
            if "layer_norms" in modules or "layernorms" in modules:
                self._set_trainable(layer.self_attn_layer_norm, trainable=False)
                self._set_trainable(layer.encoder_attn_layer_norm, trainable=False)
                self._set_trainable(layer.final_layer_norm, trainable=False)

        if "lm_head" in modules or "all" in modules:
            self._set_trainable(self.lm_head, trainable=False)
            self.final_logits_bias.requires_grad = False

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
        encoder_hidden_states = self.audio_projection(encoder_hidden_states)
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
            label_smoothing=self.label_smoothing,
        )
        return DecoderOutput(loss=loss, logits=logits)

    def sequence_losses(
        self,
        encoder_hidden_states: torch.Tensor,
        encoder_padding_mask: torch.Tensor,
        labels: torch.Tensor,
        label_smoothing: float | None = None,
    ) -> torch.Tensor:
        decoder_input_ids = shift_tokens_right(
            labels=labels,
            pad_token_id=self.config.pad_token_id,
            decoder_start_token_id=self.config.decoder_start_token_id,
        )
        decoder_attention_mask = decoder_input_ids.ne(self.config.pad_token_id).long()
        encoder_hidden_states = self.audio_projection(encoder_hidden_states)
        encoder_attention_mask = (~encoder_padding_mask).long()

        outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            return_dict=True,
        )
        logits = self.lm_head(outputs.last_hidden_state) + self.final_logits_bias
        token_losses = F.cross_entropy(
            logits.transpose(1, 2),
            labels,
            ignore_index=-100,
            reduction="none",
            label_smoothing=(
                self.label_smoothing if label_smoothing is None else label_smoothing
            ),
        )
        sequence_lengths = labels.ne(-100).sum(dim=-1).clamp_min(1)
        return token_losses.sum(dim=-1) / sequence_lengths

    @torch.no_grad()
    def score_labels(
        self,
        encoder_hidden_states: torch.Tensor,
        encoder_padding_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        decoder_input_ids = shift_tokens_right(
            labels=labels,
            pad_token_id=self.config.pad_token_id,
            decoder_start_token_id=self.config.decoder_start_token_id,
        )
        decoder_attention_mask = decoder_input_ids.ne(self.config.pad_token_id).long()
        encoder_hidden_states = self.audio_projection(encoder_hidden_states)
        encoder_attention_mask = (~encoder_padding_mask).long()

        outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            return_dict=True,
        )
        logits = self.lm_head(outputs.last_hidden_state) + self.final_logits_bias
        token_nll = F.cross_entropy(
            logits.transpose(1, 2),
            labels,
            ignore_index=-100,
            reduction="none",
        )
        sequence_lengths = labels.ne(-100).sum(dim=-1).clamp_min(1)
        return token_nll.sum(dim=-1) / sequence_lengths

    def _next_token_logits(
        self,
        generated: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        decoder_attention_mask = generated.ne(self.config.pad_token_id).long()
        outputs = self.decoder(
            input_ids=generated,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            return_dict=True,
        )
        return self.lm_head(outputs.last_hidden_state[:, -1]) + self.final_logits_bias

    def _apply_no_repeat_ngram(
        self,
        logits: torch.Tensor,
        generated: torch.Tensor,
        no_repeat_ngram_size: int,
    ) -> torch.Tensor:
        if no_repeat_ngram_size <= 0 or generated.size(1) + 1 < no_repeat_ngram_size:
            return logits

        prefix_length = no_repeat_ngram_size - 1
        for batch_index in range(generated.size(0)):
            tokens = generated[batch_index].tolist()
            current_prefix = tuple(tokens[-prefix_length:])
            banned_tokens = []

            for start in range(len(tokens) - no_repeat_ngram_size + 1):
                ngram = tuple(tokens[start : start + no_repeat_ngram_size])
                if ngram[:-1] == current_prefix:
                    banned_tokens.append(ngram[-1])

            if banned_tokens:
                logits[batch_index, banned_tokens] = -float("inf")

        return logits

    def _sample_next_token(
        self,
        logits: torch.Tensor,
        temperature: float,
        top_p: float,
    ) -> torch.Tensor:
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0 for sampling, got {temperature}")

        logits = logits / temperature

        if top_p < 1.0:
            if top_p <= 0.0:
                raise ValueError(f"top_p must be in (0, 1], got {top_p}")

            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            sorted_probs = F.softmax(sorted_logits, dim=-1)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

            remove_mask = cumulative_probs > top_p
            remove_mask[..., 1:] = remove_mask[..., :-1].clone()
            remove_mask[..., 0] = False
            sorted_logits = sorted_logits.masked_fill(remove_mask, -float("inf"))

            filtered_logits = torch.full_like(logits, -float("inf"))
            logits = filtered_logits.scatter(dim=-1, index=sorted_indices, src=sorted_logits)

        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)

    @torch.no_grad()
    def decode(
        self,
        encoder_hidden_states: torch.Tensor,
        encoder_padding_mask: torch.Tensor,
        max_length: int = 32,
        min_length: int = 0,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_p: float = 1.0,
        num_return_sequences: int = 1,
        no_repeat_ngram_size: int = 0,
    ) -> torch.Tensor:
        if num_return_sequences < 1:
            raise ValueError(
                f"num_return_sequences must be >= 1, got {num_return_sequences}"
            )

        if min_length < 0:
            raise ValueError(f"min_length must be >= 0, got {min_length}")

        encoder_hidden_states = self.audio_projection(encoder_hidden_states)

        if num_return_sequences > 1:
            encoder_hidden_states = encoder_hidden_states.repeat_interleave(
                num_return_sequences,
                dim=0,
            )
            encoder_padding_mask = encoder_padding_mask.repeat_interleave(
                num_return_sequences,
                dim=0,
            )

        batch_size = encoder_hidden_states.size(0)
        generated = torch.full(
            (batch_size, 1),
            fill_value=self.config.decoder_start_token_id,
            dtype=torch.long,
            device=encoder_hidden_states.device,
        )
        encoder_attention_mask = (~encoder_padding_mask).long()
        finished = torch.zeros(batch_size, dtype=torch.bool, device=generated.device)

        for _ in range(max_length - 1):
            logits = self._next_token_logits(
                generated=generated,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
            )

            if generated.size(1) < min_length:
                logits[:, self.config.eos_token_id] = -float("inf")

            logits = self._apply_no_repeat_ngram(
                logits=logits,
                generated=generated,
                no_repeat_ngram_size=no_repeat_ngram_size,
            )

            if do_sample:
                next_token = self._sample_next_token(
                    logits=logits,
                    temperature=temperature,
                    top_p=top_p,
                )
            else:
                next_token = logits.argmax(dim=-1, keepdim=True)

            next_token = torch.where(
                finished.unsqueeze(1),
                torch.full_like(next_token, self.config.pad_token_id),
                next_token,
            )
            generated = torch.cat([generated, next_token], dim=1)
            finished = finished | next_token.squeeze(1).eq(self.config.eos_token_id)

            if torch.all(finished):
                break

        return generated

    @torch.no_grad()
    def greedy_decode(
        self,
        encoder_hidden_states: torch.Tensor,
        encoder_padding_mask: torch.Tensor,
        max_length: int = 32,
    ) -> torch.Tensor:
        return self.decode(
            encoder_hidden_states=encoder_hidden_states,
            encoder_padding_mask=encoder_padding_mask,
            max_length=max_length,
            do_sample=False,
        )