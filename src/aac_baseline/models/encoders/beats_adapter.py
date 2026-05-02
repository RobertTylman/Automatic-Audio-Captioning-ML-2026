import torch
import torch.nn as nn
import torchaudio

from ..common import EncoderOutput, lengths_to_padding_mask
from .base import AudioEncoderBase


class MultiLayerAggregator(nn.Module):
    """
    This module implements the DCASE 2024 submission:

    1. collect hidden states from multiple encoder layers
    2. concatenate them along the feature dimension
    3. compress them back to one working hidden size

    This is intentionally easier to reason about than the 2023 repo's
    "pick one layer or learn a weighted average of layers" logic.
    """

    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, layer_outputs: list[torch.Tensor]) -> torch.Tensor:
        concatenated = torch.cat(layer_outputs, dim=-1)
        fused = self.norm(concatenated)
        fused = self.fc1(fused)
        fused = self.act(fused)
        fused = self.fc2(fused)
        return fused


class BeatsEncoderAdapter(AudioEncoderBase):
    """
    This is a scaffold-friendly BEATs-style adapter.

    Important:
    - The public interface is the part we want to keep.
    - The internal backbone is intentionally lightweight for now.

    Later, one teammate can replace the internal stack with the real BEATs
    checkpoint loader without changing the rest of the training pipeline.
    """

    def __init__(self, config: dict, sample_rate: int = 16000) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self.n_mels = config["n_mels"]
        self.hop_length = config["hop_length"]
        self.hidden_size = config["hidden_size"]

        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=config["n_fft"],
            hop_length=config["hop_length"],
            n_mels=config["n_mels"],
        )

        self.input_projection = nn.Linear(config["n_mels"], self.hidden_size)
        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=self.hidden_size,
                    nhead=config["num_heads"],
                    dim_feedforward=self.hidden_size * config["ff_multiplier"],
                    dropout=config["dropout"],
                    batch_first=True,
                    activation="gelu",
                )
                for _ in range(config["num_layers"])
            ]
        )
        self.aggregator = MultiLayerAggregator(
            input_dim=self.hidden_size * config["num_layers"],
            hidden_dim=config["aggregation_hidden_size"],
        )

    def _waveforms_to_log_mel(self, waveforms: torch.Tensor) -> torch.Tensor:
        mel = self.mel_transform(waveforms)
        mel = torch.log(mel.clamp(min=1e-5))
        return mel.transpose(1, 2)

    def _lengths_to_frame_lengths(self, waveform_lengths: torch.Tensor, max_frames: int) -> torch.Tensor:
        frame_lengths = torch.div(
            waveform_lengths + self.hop_length - 1,
            self.hop_length,
            rounding_mode="floor",
        )
        return frame_lengths.clamp(max=max_frames)

    def forward(
        self,
        waveforms: torch.Tensor,
        waveform_lengths: torch.Tensor,
    ) -> EncoderOutput:
        features = self._waveforms_to_log_mel(waveforms)
        features = self.input_projection(features)

        frame_lengths = self._lengths_to_frame_lengths(
            waveform_lengths=waveform_lengths,
            max_frames=features.size(1),
        )
        padding_mask = lengths_to_padding_mask(frame_lengths, features.size(1))

        hidden_states = []
        current = features

        for layer in self.layers:
            current = layer(current, src_key_padding_mask=padding_mask)
            hidden_states.append(current)

        aggregated = self.aggregator(hidden_states)
        return EncoderOutput(sequence=aggregated, padding_mask=padding_mask)
