import torch
import torch.nn as nn

from .common import EncoderOutput
from .decoder import BartCaptionDecoder, DecoderOutput
from .encoders import ASTEncoderAdapter, BeatsEncoderAdapter, ConvNextEncoderAdapter
from .fusion import build_fusion_module
from .postencoder import ConformerPostEncoder


class DCASE24BaselineCaptioner(nn.Module):
    """
    Top-level model wiring.

    This file is intentionally boring:
    - call BEATs (always)
    - call any other encoders that are configured
    - fuse all of them
    - refine with conformer
    - decode with BART

    Encoders other than BEATs are opt-in: they only run if their config
    block is present in the YAML. The fusion module is sized at
    construction time based on which branches are active.
    """

    def __init__(self, config: dict) -> None:
        super().__init__()
        self.config = config

        # 1. Primary encoder: BEATs (always present)
        self.beats_encoder = BeatsEncoderAdapter(config["beats_encoder"])

        # 2. Optional secondary encoders, opt-in via config presence
        self.convnext_encoder = (
            ConvNextEncoderAdapter(config["convnext_encoder"])
            if "convnext_encoder" in config
            else None
        )
        self.ast_encoder = (
            ASTEncoderAdapter(config["ast_encoder"])
            if "ast_encoder" in config
            else None
        )

        # 3. Build the fusion module sized for the active branches.
        # BEATs is the anchor (first), others follow in a fixed order.
        self.branch_names: list[str] = ["beats_encoder"]
        branch_dims: list[int] = [config["beats_encoder"]["aggregation_hidden_size"]]

        if self.convnext_encoder is not None:
            self.branch_names.append("convnext_encoder")
            branch_dims.append(config["convnext_encoder"]["hidden_size"])

        if self.ast_encoder is not None:
            self.branch_names.append("ast_encoder")
            branch_dims.append(self.ast_encoder.hidden_size)

        if len(self.branch_names) < 2:
            raise ValueError(
                "Captioner needs at least 2 encoders for fusion. "
                "Add a `convnext_encoder` or `ast_encoder` block to the config."
            )

        self.fusion = build_fusion_module(
            fusion_config=config["fusion"],
            input_dims=branch_dims,
            branch_names=self.branch_names,
        )

        self.postencoder = ConformerPostEncoder(config["conformer"])
        self.decoder = BartCaptionDecoder(config["bart"])

        self.fusion_stage = config["fusion"]["stage"]

        self.module_registry = {
            "beats_encoder": self.beats_encoder,
            "fusion": self.fusion,
            "postencoder": self.postencoder,
            "decoder": self.decoder,
        }
        if self.convnext_encoder is not None:
            self.module_registry["convnext_encoder"] = self.convnext_encoder
        if self.ast_encoder is not None:
            self.module_registry["ast_encoder"] = self.ast_encoder

    # the following section is for easily declaring which modules will be trainable
    # and which modules will be frozen, we can declare these things in the config.yml
    def set_module_trainable(self, module_name: str, trainable: bool) -> None:
        if module_name not in self.module_registry:
            raise KeyError(f"Unknown module name: {module_name}")

        for parameter in self.module_registry[module_name].parameters():
            parameter.requires_grad = trainable

    def _run_encoders(
        self,
        waveforms: torch.Tensor,
        waveform_lengths: torch.Tensor,
        sample_rates: torch.Tensor,
    ) -> list[EncoderOutput]:
        # BEATs is always the anchor (first branch).
        outputs: list[EncoderOutput] = [
            self.beats_encoder(waveforms, waveform_lengths, sample_rates)
        ]
        if self.convnext_encoder is not None:
            outputs.append(self.convnext_encoder(waveforms, waveform_lengths, sample_rates))
        if self.ast_encoder is not None:
            outputs.append(self.ast_encoder(waveforms, waveform_lengths, sample_rates))
        return outputs

    def encode_audio(
        self,
        waveforms: torch.Tensor,
        waveform_lengths: torch.Tensor,
        sample_rates: torch.Tensor,
    ) -> EncoderOutput:
        encoder_outputs = self._run_encoders(waveforms, waveform_lengths, sample_rates)

        if self.fusion_stage == "early":
            # Early fusion: combine all branches first, then refine in the conformer.
            fused_output = self.fusion(*encoder_outputs)
            return self.postencoder(fused_output)

        if self.fusion_stage == "late":
            # Late fusion: refine BEATs through the conformer first, then merge
            # with the remaining branches.
            refined_beats = self.postencoder(encoder_outputs[0])
            return self.fusion(refined_beats, *encoder_outputs[1:])

        raise ValueError(f"Unsupported fusion stage: {self.fusion_stage}")

    def forward(
        self,
        waveforms: torch.Tensor,
        waveform_lengths: torch.Tensor,
        sample_rates: torch.Tensor,
        labels: torch.Tensor,
    ) -> DecoderOutput:
        encoder_output = self.encode_audio(waveforms, waveform_lengths, sample_rates)
        return self.decoder(
            encoder_hidden_states=encoder_output.sequence,
            encoder_padding_mask=encoder_output.padding_mask,
            labels=labels,
        )

    @torch.no_grad()
    def generate(
        self,
        waveforms: torch.Tensor,
        waveform_lengths: torch.Tensor,
        sample_rates: torch.Tensor,
        max_length: int = 32,
    ) -> torch.Tensor:
        encoder_output = self.encode_audio(waveforms, waveform_lengths, sample_rates)
        return self.decoder.greedy_decode(
            encoder_hidden_states=encoder_output.sequence,
            encoder_padding_mask=encoder_output.padding_mask,
            max_length=max_length,
        )