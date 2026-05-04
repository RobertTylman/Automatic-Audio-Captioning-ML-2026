import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import EncoderOutput


class FeatureFusion(nn.Module):
    """
    Feature fusion follows the paper's idea:

    - make ConvNeXt features match the BEATs time axis
    - concatenate along the feature dimension
    - compress back to a working hidden size
    """

    def __init__(self, input_dim_a: int, input_dim_b: int, output_dim: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(input_dim_a + input_dim_b)
        self.fc1 = nn.Linear(input_dim_a + input_dim_b, output_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(output_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def _interpolate_to_length(self, sequence: torch.Tensor, target_length: int) -> torch.Tensor:
        sequence = sequence.transpose(1, 2)
        sequence = F.interpolate(sequence, size=target_length, mode="linear", align_corners=False)
        return sequence.transpose(1, 2)

    def forward(self, left: EncoderOutput, right: EncoderOutput) -> EncoderOutput:
        # In feature fusion, we want "same time step, richer feature vector".
        # If the second encoder is coarser in time, we interpolate it so both
        # branches describe the same timeline before concatenation.
        right_resampled = self._interpolate_to_length(right.sequence, left.sequence.size(1))
        fused = torch.cat([left.sequence, right_resampled], dim=-1)
        fused = self.norm(fused)
        fused = self.fc1(fused)
        fused = self.act(fused)
        fused = self.dropout(fused)
        fused = self.fc2(fused)
        return EncoderOutput(sequence=fused, padding_mask=left.padding_mask)


class SequenceFusion(nn.Module):
    """
    Sequence fusion keeps the two encoders on separate time steps and simply
    appends one sequence after the other. The decoder can then attend to both.
    """

    def __init__(self, input_dim_a: int, input_dim_b: int, output_dim: int, dropout: float) -> None:
        super().__init__()
        self.left_projection = nn.Linear(input_dim_a, output_dim)
        self.right_projection = nn.Linear(input_dim_b, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, left: EncoderOutput, right: EncoderOutput) -> EncoderOutput:
        left_sequence = self.dropout(self.left_projection(left.sequence))
        right_sequence = self.dropout(self.right_projection(right.sequence))
        sequence = torch.cat([left_sequence, right_sequence], dim=1)
        padding_mask = torch.cat([left.padding_mask, right.padding_mask], dim=1)
        return EncoderOutput(sequence=sequence, padding_mask=padding_mask)


def build_fusion_module(
    fusion_config: dict,
    left_dim: int,
    right_dim: int,
) -> nn.Module:
    if fusion_config["mode"] == "feature":
        return FeatureFusion(
            input_dim_a=left_dim,
            input_dim_b=right_dim,
            output_dim=fusion_config["output_dim"],
            dropout=fusion_config["dropout"],
        )

    if fusion_config["mode"] == "sequence":
        return SequenceFusion(
            input_dim_a=left_dim,
            input_dim_b=right_dim,
            output_dim=fusion_config["output_dim"],
            dropout=fusion_config["dropout"],
        )

    raise ValueError(f"Unsupported fusion mode: {fusion_config['mode']}")
