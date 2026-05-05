import torch
import torch.nn as nn
import torchaudio

from ..common import (
    EncoderOutput,
    downsample_lengths,
    lengths_to_padding_mask,
    resample_batch_waveforms,
)
from .base import AudioEncoderBase


class ConvNextDummyEncoder(AudioEncoderBase):
    """
    This is not a real ConvNeXt implementation.

    It exists so the whole multi-encoder pipeline can run end to end while the
    team works on the real encoder adapter in a separate file.

    Design goal:
    - output a coarser time sequence than the BEATs branch
    - mimic the need for temporal alignment during fusion
    """

    def __init__(self, config: dict, sample_rate: int = 16000) -> None:
        super().__init__()
        self.sample_rate = config.get("target_sample_rate", sample_rate)
        self.n_mels = config["n_mels"]
        self.hop_length = config["hop_length"]
        self.hidden_size = config["hidden_size"]

        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=config["n_fft"],
            hop_length=config["hop_length"],
            n_mels=config["n_mels"],
        )

        self.conv_stack = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=(1, 2), padding=1),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=(1, 2), padding=1),
            nn.GELU(),
            nn.Conv2d(128, self.hidden_size, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Dropout(config["dropout"]),
        )

    def _waveforms_to_log_mel(self, waveforms: torch.Tensor) -> torch.Tensor:
        mel = self.mel_transform(waveforms)
        return torch.log(mel.clamp(min=1e-5))

    def _lengths_to_frame_lengths(self, waveform_lengths: torch.Tensor, max_frames: int) -> torch.Tensor:
        frame_lengths = torch.div(
            waveform_lengths + self.hop_length - 1,
            self.hop_length,
            rounding_mode="floor",
        )

        # Two stride-2 convolutions reduce the time axis by roughly 4x.
        frame_lengths = downsample_lengths(frame_lengths, stride=2)
        frame_lengths = downsample_lengths(frame_lengths, stride=2)
        return frame_lengths.clamp(max=max_frames)

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

        mel = self._waveforms_to_log_mel(waveforms).unsqueeze(1)
        features = self.conv_stack(mel)

        # Collapse the frequency axis so this branch also returns a sequence.
        features = features.mean(dim=2).transpose(1, 2)

        frame_lengths = self._lengths_to_frame_lengths(
            waveform_lengths=waveform_lengths,
            max_frames=features.size(1),
        )
        padding_mask = lengths_to_padding_mask(frame_lengths, features.size(1))

        return EncoderOutput(sequence=features, padding_mask=padding_mask)