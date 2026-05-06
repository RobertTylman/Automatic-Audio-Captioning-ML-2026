import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import EncoderOutput


class FeatureFusion(nn.Module):
    """
    Feature fusion follows the paper's idea:

    - make the secondary branches match the primary (first) branch's time axis
    - concatenate along the feature dimension
    - compress back to a working hidden size

    Generalizes to any number of branches >= 2. The first branch is the
    "anchor" whose time axis everything else aligns to.
    """

    def __init__(self, input_dims: list[int], output_dim: int, dropout: float) -> None:
        super().__init__()
        if len(input_dims) < 2:
            raise ValueError(f"FeatureFusion needs >= 2 branches, got {len(input_dims)}")
        self.input_dims = list(input_dims)
        total_dim = sum(self.input_dims)
        self.norm = nn.LayerNorm(total_dim)
        self.fc1 = nn.Linear(total_dim, output_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(output_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _interpolate_to_length(sequence: torch.Tensor, target_length: int) -> torch.Tensor:
        if sequence.size(1) == target_length:
            return sequence
        sequence = sequence.transpose(1, 2)
        sequence = F.interpolate(sequence, size=target_length, mode="linear", align_corners=False)
        return sequence.transpose(1, 2)

    def forward(self, *branches: EncoderOutput) -> EncoderOutput:
        if len(branches) != len(self.input_dims):
            raise ValueError(
                f"FeatureFusion was built for {len(self.input_dims)} branches, "
                f"got {len(branches)}"
            )

        anchor = branches[0]
        target_length = anchor.sequence.size(1)

        aligned_sequences = [anchor.sequence]
        for branch in branches[1:]:
            aligned_sequences.append(
                self._interpolate_to_length(branch.sequence, target_length)
            )

        fused = torch.cat(aligned_sequences, dim=-1)
        fused = self.norm(fused)
        fused = self.fc1(fused)
        fused = self.act(fused)
        fused = self.dropout(fused)
        fused = self.fc2(fused)
        return EncoderOutput(sequence=fused, padding_mask=anchor.padding_mask)


class SequenceFusion(nn.Module):
    """
    Sequence fusion keeps each encoder on its own time steps and simply
    appends the sequences end-to-end. The decoder can then attend to all
    branches as one longer sequence.

    Generalizes to any number of branches >= 2.
    """

    def __init__(self, input_dims: list[int], output_dim: int, dropout: float) -> None:
        super().__init__()
        if len(input_dims) < 2:
            raise ValueError(f"SequenceFusion needs >= 2 branches, got {len(input_dims)}")
        self.input_dims = list(input_dims)
        self.projections = nn.ModuleList(
            [nn.Linear(input_dim, output_dim) for input_dim in self.input_dims]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, *branches: EncoderOutput) -> EncoderOutput:
        if len(branches) != len(self.projections):
            raise ValueError(
                f"SequenceFusion was built for {len(self.projections)} branches, "
                f"got {len(branches)}"
            )

        projected_sequences = [
            self.dropout(projection(branch.sequence))
            for projection, branch in zip(self.projections, branches)
        ]
        sequence = torch.cat(projected_sequences, dim=1)
        padding_mask = torch.cat([branch.padding_mask for branch in branches], dim=1)
        return EncoderOutput(sequence=sequence, padding_mask=padding_mask)


def build_fusion_module(
    fusion_config: dict,
    input_dims: list[int],
) -> nn.Module:
    if fusion_config["mode"] == "feature":
        return FeatureFusion(
            input_dims=input_dims,
            output_dim=fusion_config["output_dim"],
            dropout=fusion_config["dropout"],
        )

    if fusion_config["mode"] == "sequence":
        return SequenceFusion(
            input_dims=input_dims,
            output_dim=fusion_config["output_dim"],
            dropout=fusion_config["dropout"],
        )

    raise ValueError(f"Unsupported fusion mode: {fusion_config['mode']}")
