import logging
import random
from typing import Optional, List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import ModelOutput
from transformers.models.bart.modeling_bart import (
    BartConfig,
    BartDecoder,
    BartPretrainedModel,
    BartLearnedPositionalEmbedding,
    Seq2SeqLMOutput,
    shift_tokens_right,
    logger,
)
from transformers.models.wav2vec2_conformer.modeling_wav2vec2_conformer import (
    Wav2Vec2ConformerConfig,
    Wav2Vec2ConformerEncoder,
)
from info_nce import InfoNCE

from BEATs import BEATs, BEATsConfig
from ConvNext import ConvNextEncoder
from modules import GradMultiply
from specaug import SpecAug


class BidirectionalInfoNCELoss(nn.Module):
    def __init__(self, temperature=0.07) -> None:
        super().__init__()
        self.loss_func = InfoNCE(temperature=temperature, negative_mode="paired")

    def forward(self, audio_embeds, text_embeds):
        assert audio_embeds.size() == text_embeds.size()
        batch_size = audio_embeds.size(0)

        text_negatives = []
        audio_negatives = []

        for i in range(batch_size):
            text_negatives.append(
                torch.cat([text_embeds[:i], text_embeds[i + 1 :]], dim=0)
            )

            audio_negatives.append(
                torch.cat([audio_embeds[:i], audio_embeds[i + 1 :]], dim=0)
            )

        text_negatives = torch.stack(text_negatives, dim=0)
        audio_negatives = torch.stack(audio_negatives, dim=0)

        total_loss = 0.5 * (
            self.loss_func(audio_embeds, text_embeds, text_negatives)
            + self.loss_func(text_embeds, audio_embeds, audio_negatives)
        )

        return total_loss


class RelationalDistilCosineMSELoss(nn.Module):
    def __init__(self, eps=1e-8) -> None:
        super().__init__()
        self.eps = eps
        self.loss_func = nn.MSELoss()

    def forward(self, audio_embeds, text_embeds):
        assert audio_embeds.size() == text_embeds.size()

        audio_norms = audio_embeds.norm(dim=1)[:, None]
        text_norms = text_embeds.norm(dim=1)[:, None]

        audio_embeds = audio_embeds / torch.clamp(audio_norms, min=self.eps)
        text_embeds = text_embeds / torch.clamp(text_norms, min=self.eps)

        audio_ssm = torch.mm(audio_embeds, audio_embeds.transpose(0, 1))
        text_ssm = torch.mm(text_embeds, text_embeds.transpose(0, 1))

        return self.loss_func(audio_ssm, text_ssm)


class TextEmbeddingPredictor(nn.Module):
    def __init__(self, config: BartConfig):
        super().__init__()

        self.fc1 = nn.Linear(config.d_model, config.embed_predictor_ffn_dim, bias=False)
        self.bn = nn.BatchNorm1d(config.embed_predictor_ffn_dim)
        self.act_fn = nn.GELU()
        self.fc2 = nn.Linear(
            config.embed_predictor_ffn_dim, config.embed_predictor_out_dim, bias=False
        )

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn(x)
        x = self.act_fn(x)
        x = self.fc2(x)

        return x


from transformers.generation.utils import GenerationMixin

class FusedEncodersConformerBartSeq2SeqForCaptioning(BartPretrainedModel, GenerationMixin):
    def __init__(
        self,
        config: BartConfig,
        conformer_config: Wav2Vec2ConformerConfig,
        beats_config: BEATsConfig = None,
        for_inference: bool = False,
    ):
        super().__init__(config)
        
        # 1. Primary Encoder: BEATs
        if beats_config is not None:
            self.encoder = BEATs(beats_config)
        elif getattr(config, "pretrained_beats_path", None) is not None:
            print("[info] loading BEATs config from ckpt")
            self.encoder = BEATs(
                BEATsConfig(torch.load(config.pretrained_beats_path)["cfg"])
            )
        else:
            raise ValueError(
                "provide either `beats_config` " "or `config.pretrained_beats_path`"
            )

        # 2. Secondary Encoder: ConvNeXt
        self.convnext = ConvNextEncoder()

        # 3. Aggregation MLP for Fusion
        beats_dim = self.encoder.embed if hasattr(self.encoder, 'embed') else 768
        # BEATs output is actually encoder_embed_dim = 768 usually
        beats_dim = 768
        convnext_dim = self.convnext.hidden_size
        fused_dim = beats_dim + convnext_dim
        
        self.fusion_mlp = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, 3072),
            nn.GELU(),
            nn.Linear(3072, conformer_config.hidden_size)
        )

        self.postencoder = Wav2Vec2ConformerEncoder(conformer_config)

        self.main_input_name = "decoder_input_ids"
        self.for_inference = for_inference

        self.decoder = BartDecoder(config)
        self.register_parameter(
            "final_logits_bias", nn.Parameter(torch.zeros((1, config.vocab_size)))
        )
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.cross_embed_positions = BartLearnedPositionalEmbedding(
            config.max_cross_position_embeddings, config.d_model
        )

        self.lsm_weight = config.lsm_weight
        self.use_weighted_encoder_repr = config.use_weighted_encoder_repr

        if self.use_weighted_encoder_repr:
            self.register_parameter(
                "encoder_repr_layer_weights",
                nn.Parameter(torch.ones((config.encoder_layers + 1, 1))),
            )
            self.encoder_repr_layer_idx = None
        else:
            self.encoder_repr_layer_weights = None
            self.encoder_repr_layer_idx = getattr(
                config, "encoder_repr_layer_idx", config.encoder_layers
            )

        if config.d_model != conformer_config.hidden_size:
            self.enc_dec_proj = nn.Linear(
                conformer_config.hidden_size, config.d_model, bias=False
            )
        else:
            self.enc_dec_proj = None

        # freeze beats and convnext encoders
        if getattr(config, "freeze_encoder", True):
            for param in self.encoder.parameters():
                param.requires_grad_(False)
            for param in self.convnext.parameters():
                param.requires_grad_(False)

        # downsample encoder representations
        self.encoder_downsample_rate = 1
        self.downsample_conv = None

        if getattr(config, "encoder_downsample_rate", None) is not None:
            ds_rate = int(config.encoder_downsample_rate)
            if ds_rate < 1:
                raise ValueError(
                    "`encoder_downsample_rate` should be >= 1, " f"got {ds_rate}"
                )
            self.encoder_downsample_rate = ds_rate

        if getattr(config, "use_conv_downsample", False):
            ds_rate = getattr(config, "encoder_downsample_rate", 1)
            kernel_multiplier = getattr(config, "downsample_kernel_size", 1.5)
            self.downsample_conv = nn.Conv1d(
                in_channels=conformer_config.hidden_size,
                out_channels=conformer_config.hidden_size,
                kernel_size=int(round(ds_rate * kernel_multiplier)),
                stride=ds_rate,
            )
            self.encoder_downsample_rate = ds_rate

        # For encoder multilabel keyword recognition
        self.encoder_clf_proj = None
        if getattr(config, "use_encoder_clf", False):
            n_enc_clf_classes = getattr(config, "n_enc_clf_classes", 2609)
            self.encoder_clf_proj = nn.Linear(config.d_model, n_enc_clf_classes)

        self.encoder_clf_loss_weight = getattr(config, "encoder_clf_loss_weight", 1.0)

        # For encoder text embedding prediction
        self.encoder_embed_mlp = None
        if getattr(config, "use_encoder_embed_mlp", False):
            assert (
                self.encoder_clf_proj is None,
                "keyword clf & text embed predictor should not be used together",
            )
            self.encoder_embed_mlp = TextEmbeddingPredictor(config)

        # Use InfoNCE on text embedding prediction, instead of cosine sim
        self.use_contrastive_embed_loss = getattr(config, "use_contrastive_embed_loss", False)
        self.contrastive_temperature = getattr(config, "contrastive_temperature", 0.07)

        # Use Relational distillation, instead of cosine sim or InfoNCE
        self.use_relational_embed_distil = getattr(config, "use_relational_embed_distil", False)

        if self.use_relational_embed_distil:
            assert not self.use_contrastive_embed_loss

        if self.use_contrastive_embed_loss or self.use_relational_embed_distil:
            assert self.encoder_embed_mlp is not None

        self.embed_predictor_loss_weight = 5.0 if not self.use_contrastive_embed_loss else 1.0
        if hasattr(config, "embed_predictor_loss_weight"):
            self.embed_predictor_loss_weight = config.embed_predictor_loss_weight

        # Initialize weights and apply final processing
        self.post_init()

        if getattr(config, "pretrained_beats_path", None) is not None and not for_inference:
            self.load_encoder_pretrained_weights(config.pretrained_beats_path)

        self.spec_aug = None
        if getattr(config, "spec_aug", None) is not None and not for_inference:
            self.add_encoder_spec_aug(config.spec_aug)
            self.add_spec_aug(config.spec_aug)

        self.encoder_grad_multiplier = getattr(config, "encoder_grad_multiplier", None)
        self.randomize_audio_feats = getattr(config, "randomize_audio_feats", False)

    def forward_padding_mask(self, features: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        extra = padding_mask.size(1) % features.size(1)
        if extra > 0:
            padding_mask = padding_mask[:, :-extra]
        padding_mask = padding_mask.view(padding_mask.size(0), features.size(1), -1)
        padding_mask = padding_mask.all(-1)
        return padding_mask

    def load_encoder_pretrained_weights(self, ckpt_path):
        self.encoder.load_state_dict(torch.load(ckpt_path)["model"])
        print("[info] pretrained BEATs weights loaded")

    def add_spec_aug(self, spec_aug_config):
        self.spec_aug = SpecAug(**spec_aug_config)

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def get_encoder_repr_layer_weights(self):
        if self.use_weighted_encoder_repr:
            return nn.functional.softmax(self.encoder_repr_layer_weights.squeeze())
        return None

    def add_encoder_spec_aug(self, spec_aug_config):
        self.encoder.add_spec_aug(spec_aug_config)
        print("[info] SpecAug added to BEATs encoder")

    def get_input_embeddings(self):
        return self.decoder.embed_tokens

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def run_encoder_clf(self, encoder_hidden_states, encoder_attention_mask):
        encoder_hidden_states_mean = encoder_hidden_states.sum(dim=-2)
        encoder_hidden_states_mean /= encoder_attention_mask.sum(dim=-1).unsqueeze(1)
        clf_logits = self.encoder_clf_proj(encoder_hidden_states_mean)
        return clf_logits

    def run_encoder_mlp(self, encoder_hidden_states, encoder_attention_mask):
        encoder_hidden_states_mean = encoder_hidden_states.sum(dim=-2)
        encoder_hidden_states_mean /= encoder_attention_mask.sum(dim=-1).unsqueeze(1)
        mlp_logits = self.encoder_embed_mlp(encoder_hidden_states_mean)
        return mlp_logits

    def encode_audio(self, encoder_input, attention_mask):
        # 1. Run BEATs
        encoder_outputs = self.encoder(
            source=encoder_input,
            padding_mask=attention_mask,
            max_layer=self.encoder_repr_layer_idx,
        )
        encoder_hidden_states = encoder_outputs.hidden_states

        if self.use_weighted_encoder_repr:
            repr_layer_weights = nn.functional.softmax(self.encoder_repr_layer_weights, dim=-2)
            pooled_encoder_hidden_states = torch.stack(encoder_hidden_states, dim=-2) * repr_layer_weights
            beats_features = pooled_encoder_hidden_states.sum(dim=-2)
        else:
            beats_features = encoder_hidden_states[self.encoder_repr_layer_idx]

        # 2. Run ConvNeXt
        with torch.no_grad():
            fbank = self.encoder.preprocess(encoder_input)
        convnext_features = self.convnext(fbank)
        
        # 3. Align Sequence Lengths
        # convnext_features: [B, T_c, D_c] -> [B, D_c, T_c]
        convnext_features = convnext_features.transpose(1, 2)
        target_seq_len = beats_features.size(1)
        
        # Interpolate to match BEATs sequence length
        convnext_features = F.interpolate(
            convnext_features, 
            size=target_seq_len, 
            mode='linear', 
            align_corners=False
        )
        # [B, D_c, T_b] -> [B, T_b, D_c]
        convnext_features = convnext_features.transpose(1, 2)

        # 4. Feature Concatenation
        # [B, T_b, D_b + D_c]
        fused_features = torch.cat([beats_features, convnext_features], dim=-1)
        
        # 5. Aggregation MLP
        fused_features = self.fusion_mlp(fused_features)

        # Continue with downsampling and Conformer as in the original code
        if self.encoder_downsample_rate > 1 and self.downsample_conv is None:
            offset = random.choice(range(self.encoder_downsample_rate))
            fused_features = fused_features[:, offset :: self.encoder_downsample_rate]
            encoder_outputs.attention_mask = encoder_outputs.attention_mask[:, offset :: self.encoder_downsample_rate]
        elif self.downsample_conv is not None:
            fused_features = self.downsample_conv(fused_features.transpose(1, 2)).transpose(1, 2)
            encoder_outputs.attention_mask = self.forward_padding_mask(fused_features, encoder_outputs.attention_mask)

        if self.spec_aug is not None and self.training:
            fused_features = self.spec_aug(fused_features)[0]

        if encoder_outputs.attention_mask is not None:
            conformer_attn_mask = ~encoder_outputs.attention_mask
        else:
            conformer_attn_mask = None

        # run through conformer
        pooled_encoder_hidden_states = self.postencoder(
            fused_features,
            attention_mask=conformer_attn_mask,
        ).last_hidden_state

        if encoder_outputs.attention_mask is not None:
            assert pooled_encoder_hidden_states.size(1) == encoder_outputs.attention_mask.size(1)

        return ModelOutput(
            last_hidden_state=pooled_encoder_hidden_states,
            hidden_states=None,
            attentions=None,
            attention_mask=encoder_outputs.attention_mask,
        )

    def forward(
        self,
        encoder_input: torch.FloatTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_labels: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[List[torch.FloatTensor]] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, Seq2SeqLMOutput]:
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            if use_cache:
                logger.warning("The `use_cache` argument is changed to `False` since `labels` is provided.")
            use_cache = False
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        if encoder_outputs is None:
            encoder_outputs = self.encode_audio(
                encoder_input,
                attention_mask,
            )
            attention_mask = encoder_outputs.attention_mask

        pooled_encoder_hidden_states = encoder_outputs.last_hidden_state

        if self.randomize_audio_feats:
            pooled_encoder_hidden_states = torch.randn_like(pooled_encoder_hidden_states)

        if self.encoder_grad_multiplier is not None:
            pooled_encoder_hidden_states = GradMultiply.apply(
                pooled_encoder_hidden_states, self.encoder_grad_multiplier
            )

        if attention_mask is not None:
            attention_mask = (~attention_mask).float()

        if self.enc_dec_proj is not None:
            pooled_encoder_hidden_states = self.enc_dec_proj(pooled_encoder_hidden_states)

        if self.encoder_clf_proj is not None:
            encoder_clf_logits = self.run_encoder_clf(pooled_encoder_hidden_states, attention_mask)

        if self.encoder_embed_mlp is not None:
            encoder_mlp_logits = self.run_encoder_mlp(pooled_encoder_hidden_states, attention_mask)

        pooled_encoder_hidden_states = (
            pooled_encoder_hidden_states
            + self.cross_embed_positions(pooled_encoder_hidden_states)
        )

        outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=pooled_encoder_hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        lm_logits = self.lm_head(outputs[0])
        lm_logits = lm_logits + self.final_logits_bias.to(lm_logits.device)

        masked_lm_loss = None
        total_loss = None
        if labels is not None:
            if not self.for_inference:
                loss_fct = nn.CrossEntropyLoss(label_smoothing=self.lsm_weight)
                masked_lm_loss = loss_fct(
                    lm_logits.view(-1, self.config.vocab_size), labels.view(-1)
                )
                total_loss = masked_lm_loss
            else:
                loss_fct = nn.CrossEntropyLoss(reduction="none")
                masked_lm_loss = loss_fct(lm_logits.permute(0, 2, 1), labels)
                masked_lm_loss = masked_lm_loss.sum(dim=-1)
                seqlens = torch.count_nonzero(labels != -100, dim=-1).float()
                masked_lm_loss = masked_lm_loss / seqlens
                total_loss = masked_lm_loss

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((total_loss,) + output) if total_loss is not None else output

        return Seq2SeqLMOutput(
            loss=total_loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.hidden_states,
            decoder_attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
        )

    def prepare_inputs_for_generation(self, decoder_input_ids, past_key_values=None, attention_mask=None,
                                      decoder_attention_mask=None, head_mask=None, decoder_head_mask=None,
                                      cross_attn_head_mask=None, use_cache=None, encoder_outputs=None,
                                      encoder_input=None, **kwargs):
        if past_key_values is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        return {
            "encoder_input": encoder_input,
            "encoder_outputs": encoder_outputs,
            "past_key_values": past_key_values,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "decoder_attention_mask": decoder_attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,
        }

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return shift_tokens_right(
            labels, self.config.pad_token_id, self.config.decoder_start_token_id
        )

    @staticmethod
    def _reorder_cache(past, beam_idx):
        reordered_past = ()
        for layer_past in past:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past[:2])
                + layer_past[2:],
            )
        return reordered_past
