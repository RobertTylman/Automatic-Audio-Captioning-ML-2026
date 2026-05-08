import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import EncoderOutput


def _default_branch_names(count: int) -> list[str]:
    return [f"branch_{index}" for index in range(count)]


def _log_branch_shapes_once(
    module_name: str,
    branch_names: list[str],
    branches: tuple[EncoderOutput, ...],
    enabled: bool,
) -> None:
    if not enabled:
        return

    print(f"[fusion:{module_name}] input branch shapes", flush=True)
    for branch_name, branch in zip(branch_names, branches):
        valid_lengths = (~branch.padding_mask).sum(dim=1)
        min_valid = int(valid_lengths.min().item()) if valid_lengths.numel() else 0
        max_valid = int(valid_lengths.max().item()) if valid_lengths.numel() else 0
        print(
            f"[fusion:{module_name}] {branch_name}: "
            f"sequence={tuple(branch.sequence.shape)} "
            f"padding_mask={tuple(branch.padding_mask.shape)} "
            f"valid_length_min={min_valid} valid_length_max={max_valid}",
            flush=True,
        )


class FeatureFusion(nn.Module):
    """
    Feature fusion follows the paper's idea:

    - make the secondary branches match the primary (first) branch's time axis
    - concatenate along the feature dimension
    - compress back to a working hidden size

    Generalizes to any number of branches >= 2. The first branch is the
    "anchor" whose time axis everything else aligns to.
    """

    def __init__(
        self,
        input_dims: list[int],
        output_dim: int,
        dropout: float,
        branch_names: list[str] | None = None,
        debug_shapes: bool = False,
    ) -> None:
        super().__init__()
        if len(input_dims) < 2:
            raise ValueError(f"FeatureFusion needs >= 2 branches, got {len(input_dims)}")
        self.input_dims = list(input_dims)
        self.branch_names = branch_names or _default_branch_names(len(input_dims))
        self.debug_shapes = debug_shapes
        self._logged_debug_shapes = False
        total_dim = sum(self.input_dims)
        self.norm = nn.LayerNorm(total_dim)
        self.fc1 = nn.Linear(total_dim, output_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(output_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _interpolate_valid_prefix(sequence: torch.Tensor, valid_length: int, target_length: int) -> torch.Tensor:
        """
        Resample only the real prefix of a sequence.

        This keeps padded tail embeddings from influencing the interpolation.
        The caller is responsible for re-padding after alignment.
        """
        if target_length <= 0:
            return sequence.new_zeros((0, sequence.size(-1)))

        if valid_length <= 0:
            return sequence.new_zeros((target_length, sequence.size(-1)))

        trimmed = sequence[:valid_length]
        if valid_length == target_length:
            return trimmed

        trimmed = trimmed.transpose(0, 1).unsqueeze(0)
        resized = F.interpolate(trimmed, size=target_length, mode="linear", align_corners=False)
        return resized.squeeze(0).transpose(0, 1)

    def forward(self, *branches: EncoderOutput) -> EncoderOutput:
        if len(branches) != len(self.input_dims):
            raise ValueError(
                f"FeatureFusion was built for {len(self.input_dims)} branches, "
                f"got {len(branches)}"
            )

        _log_branch_shapes_once(
            module_name=self.__class__.__name__,
            branch_names=self.branch_names,
            branches=branches,
            enabled=self.debug_shapes and not self._logged_debug_shapes,
        )
        self._logged_debug_shapes = True

        anchor = branches[0]
        anchor_valid_lengths = (~anchor.padding_mask).sum(dim=1)
        max_target_length = anchor.sequence.size(1)
        total_dim = sum(self.input_dims)

        fused_inputs = anchor.sequence.new_zeros(
            anchor.sequence.size(0), max_target_length, total_dim
        )

        for batch_index in range(anchor.sequence.size(0)):
            target_length = int(anchor_valid_lengths[batch_index].item())
            if target_length <= 0:
                continue

            aligned_sequences = [
                anchor.sequence[batch_index, :target_length]
            ]
            for branch in branches[1:]:
                valid_length = int((~branch.padding_mask[batch_index]).sum().item())
                aligned_sequences.append(
                    self._interpolate_valid_prefix(
                        branch.sequence[batch_index],
                        valid_length=valid_length,
                        target_length=target_length,
                    )
                )

            fused_inputs[batch_index, :target_length] = torch.cat(aligned_sequences, dim=-1)

        fused = fused_inputs
        fused = self.norm(fused)
        fused = self.fc1(fused)
        fused = self.act(fused)
        fused = self.dropout(fused)
        fused = self.fc2(fused)
        fused = fused.masked_fill(anchor.padding_mask.unsqueeze(-1), 0.0)
        return EncoderOutput(sequence=fused, padding_mask=anchor.padding_mask)


class SequenceFusion(nn.Module):
    """
    Sequence fusion keeps each encoder on its own time steps and simply
    appends the sequences end-to-end. The decoder can then attend to all
    branches as one longer sequence.

    Generalizes to any number of branches >= 2.
    """

    def __init__(
        self,
        input_dims: list[int],
        output_dim: int,
        dropout: float,
        branch_names: list[str] | None = None,
        debug_shapes: bool = False,
    ) -> None:
        super().__init__()
        if len(input_dims) < 2:
            raise ValueError(f"SequenceFusion needs >= 2 branches, got {len(input_dims)}")
        self.input_dims = list(input_dims)
        self.branch_names = branch_names or _default_branch_names(len(input_dims))
        self.debug_shapes = debug_shapes
        self._logged_debug_shapes = False
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

        _log_branch_shapes_once(
            module_name=self.__class__.__name__,
            branch_names=self.branch_names,
            branches=branches,
            enabled=self.debug_shapes and not self._logged_debug_shapes,
        )
        self._logged_debug_shapes = True

        projected_sequences = [
            self.dropout(projection(branch.sequence))
            for projection, branch in zip(self.projections, branches)
        ]
        sequence = torch.cat(projected_sequences, dim=1)
        padding_mask = torch.cat([branch.padding_mask for branch in branches], dim=1)
        return EncoderOutput(sequence=sequence, padding_mask=padding_mask)


class LearnedResamplingFusion(nn.Module):
    """
    Learned alignment without interpolating latent vectors.

    The first branch defines the output time axis. Each secondary branch keeps
    its native sequence length, and anchor timesteps query it through
    cross-attention to produce an aligned representation.
    """

    def __init__(
        self,
        input_dims: list[int],
        output_dim: int,
        dropout: float,
        num_heads: int,
        branch_names: list[str] | None = None,
        debug_shapes: bool = False,
    ) -> None:
        super().__init__()
        if len(input_dims) < 2:
            raise ValueError(f"LearnedResamplingFusion needs >= 2 branches, got {len(input_dims)}")
        if output_dim % num_heads != 0:
            raise ValueError(
                "LearnedResamplingFusion output_dim must be divisible by num_heads. "
                f"Got output_dim={output_dim}, num_heads={num_heads}."
            )

        self.input_dims = list(input_dims)
        self.branch_names = branch_names or _default_branch_names(len(input_dims))
        self.debug_shapes = debug_shapes
        self._logged_debug_shapes = False
        self.projections = nn.ModuleList(
            [nn.Linear(input_dim, output_dim) for input_dim in self.input_dims]
        )
        self.resamplers = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    embed_dim=output_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    batch_first=True,
                )
                for _ in self.input_dims[1:]
            ]
        )

        total_dim = output_dim * len(self.input_dims)
        self.norm = nn.LayerNorm(total_dim)
        self.fc1 = nn.Linear(total_dim, output_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(output_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, *branches: EncoderOutput) -> EncoderOutput:
        if len(branches) != len(self.projections):
            raise ValueError(
                f"LearnedResamplingFusion was built for {len(self.projections)} branches, "
                f"got {len(branches)}"
            )

        _log_branch_shapes_once(
            module_name=self.__class__.__name__,
            branch_names=self.branch_names,
            branches=branches,
            enabled=self.debug_shapes and not self._logged_debug_shapes,
        )
        self._logged_debug_shapes = True

        projected_branches = [
            projection(branch.sequence)
            for projection, branch in zip(self.projections, branches)
        ]
        anchor = projected_branches[0]

        aligned_sequences = [anchor]
        for resampler, projected_branch, branch in zip(
            self.resamplers,
            projected_branches[1:],
            branches[1:],
        ):
            aligned_branch, _ = resampler(
                query=anchor,
                key=projected_branch,
                value=projected_branch,
                key_padding_mask=branch.padding_mask,
                need_weights=False,
            )
            aligned_sequences.append(aligned_branch)

        fused = torch.cat(aligned_sequences, dim=-1)
        fused = self.norm(fused)
        fused = self.fc1(fused)
        fused = self.act(fused)
        fused = self.dropout(fused)
        fused = self.fc2(fused)
        fused = fused.masked_fill(branches[0].padding_mask.unsqueeze(-1), 0.0)
        return EncoderOutput(sequence=fused, padding_mask=branches[0].padding_mask)


def build_fusion_module(
    fusion_config: dict,
    input_dims: list[int],
    branch_names: list[str] | None = None,
) -> nn.Module:
    debug_shapes = fusion_config.get("debug_shapes", False)
    if fusion_config["mode"] == "feature":
        return FeatureFusion(
            input_dims=input_dims,
            output_dim=fusion_config["output_dim"],
            dropout=fusion_config["dropout"],
            branch_names=branch_names,
            debug_shapes=debug_shapes,
        )

    if fusion_config["mode"] == "sequence":
        return SequenceFusion(
            input_dims=input_dims,
            output_dim=fusion_config["output_dim"],
            dropout=fusion_config["dropout"],
            branch_names=branch_names,
            debug_shapes=debug_shapes,
        )

    if fusion_config["mode"] in {"learned_resampling", "learned_resample", "cross_attention_resample"}:
        return LearnedResamplingFusion(
            input_dims=input_dims,
            output_dim=fusion_config["output_dim"],
            dropout=fusion_config["dropout"],
            num_heads=fusion_config.get("resampling_num_heads", 8),
            branch_names=branch_names,
            debug_shapes=debug_shapes,
        )

    raise ValueError(f"Unsupported fusion mode: {fusion_config['mode']}")