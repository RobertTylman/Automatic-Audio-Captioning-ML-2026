import torch

from ..common import EncoderOutput, lengths_to_padding_mask, resample_batch_waveforms
from .base import AudioEncoderBase
from .panns_convnext import convnext_tiny


class ConvNextEncoderAdapter(AudioEncoderBase):
    """
    Adapter for the audio-native ConvNeXt-Tiny backbone used with the
    AudioSet checkpoint. The preprocessing and audio stem follow the source
    repo so we stay compatible with the distributed weights.
    """

    def __init__(self, config: dict, sample_rate: int = 32000) -> None:
        super().__init__()
        self.sample_rate = config.get("target_sample_rate", sample_rate)
        self.hidden_size = config["hidden_size"]
        self.hop_size = config.get("hop_size", 320)
        self.after_stem_dim = tuple(config.get("after_stem_dim", [252, 56]))
        self.enable_spec_augment = config.get("enable_spec_augment", False)

        checkpoint_path = config.get("pretrained_checkpoint_path")
        self.model = convnext_tiny(
            pretrained_path=checkpoint_path,
            strict=config.get("strict_checkpoint_load", False),
            drop_path_rate=config.get("drop_path_rate", 0.0),
            after_stem_dim=self.after_stem_dim,
            sample_rate=self.sample_rate,
            hop_size=self.hop_size,
            mel_bins=config.get("mel_bins", 224),
            fmin=config.get("fmin", 50),
            fmax=config.get("fmax", 14000),
            enable_spec_augment=self.enable_spec_augment,
        )

        self.audio_stem_spec = getattr(self.model, "audio_stem_spec", None)
        self.checkpoint_load_result = getattr(self.model, "checkpoint_load_result", None)

    def _conv_output_lengths(
        self,
        lengths: torch.Tensor,
        kernel_size: int,
        stride: int,
        padding: int,
    ) -> torch.Tensor:
        return torch.div(
            lengths + 2 * padding - kernel_size,
            stride,
            rounding_mode="floor",
        ) + 1

    def _frame_lengths_to_output_lengths(self, frame_lengths: torch.Tensor) -> torch.Tensor:
        if self.audio_stem_spec is None:
            raise RuntimeError("ConvNeXt audio stem spec was not initialized.")

        frame_lengths = self._conv_output_lengths(
            frame_lengths,
            kernel_size=self.audio_stem_spec["kernel_size"],
            stride=self.audio_stem_spec["stride"],
            padding=self.audio_stem_spec["padding"],
        )

        for _ in range(3):
            frame_lengths = self._conv_output_lengths(
                frame_lengths,
                kernel_size=2,
                stride=2,
                padding=0,
            )

        return frame_lengths

    def forward(
        self,
        waveforms: torch.Tensor,
        waveform_lengths: torch.Tensor,
        sample_rates: torch.Tensor,
    ) -> EncoderOutput:
        waveforms, waveform_lengths = resample_batch_waveforms(
            waveforms=waveforms,
            waveform_lengths=waveform_lengths,
            sample_rates=sample_rates,
            target_sample_rate=self.sample_rate,
        )

        features = self.model.forward_frame_embeddings(waveforms)
        features = features.mean(dim=3).transpose(1, 2)

        frame_lengths = (waveform_lengths // self.hop_size) + 1
        frame_lengths = self._frame_lengths_to_output_lengths(frame_lengths)
        frame_lengths = frame_lengths.clamp(min=0, max=features.size(1))

        padding_mask = lengths_to_padding_mask(frame_lengths, features.size(1))
        return EncoderOutput(sequence=features, padding_mask=padding_mask)
