import torch
import torch.nn as nn

from ..common import EncoderOutput, resample_batch_waveforms
from .base import AudioEncoderBase
from .beats_vendor import BEATs, BEATsConfig


class ConcatThenCompressAggregator(nn.Module):
    """
    This is the exact aggregation pattern described in the DCASE 2024 paper:

    1. collect several BEATs hidden layers
    2. concatenate them along the feature dimension
    3. compress them with LayerNorm -> Linear -> GELU -> Linear

    We keep this module separate so the 2024 aggregation idea stays easy to
    inspect without getting inde the BEATs implementation.
    """

    def __init__(self, input_dim: int, output_dim: int, intermediate_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, intermediate_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(intermediate_dim, output_dim)

    def forward(self, layer_outputs: list[torch.Tensor]) -> torch.Tensor:
        concatenated = torch.cat(layer_outputs, dim=-1)
        fused = self.norm(concatenated)
        fused = self.fc1(fused)
        fused = self.act(fused)
        fused = self.fc2(fused)
        return fused


class BeatsEncoderAdapter(AudioEncoderBase):
    """
    Clean wrapper around the real BEATs implementation used by the captioning
    baseline repo.

    Important distinction:
    - the heavy BEATs code lives in `beats_vendor/`
    - this adapter owns the project-facing interface
    - the adapter also owns the 2024 layer aggregation pipeline

    That separation lets us keep the scaffold readable while still using the
    exact BEATs preprocessing and hidden-state behavior from the prior system.
    """

    def __init__(self, config: dict, sample_rate: int = 16000) -> None:
        super().__init__()
        self.sample_rate = config.get("target_sample_rate", sample_rate)
        self.config = config

        checkpoint_path = config.get("pretrained_checkpoint_path")
        checkpoint = None
        if checkpoint_path:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            beats_config = BEATsConfig(checkpoint["cfg"])
        else:
            beats_config = BEATsConfig(config["beats_config"])

        self.beats = BEATs(beats_config)

        if checkpoint is not None:
            self.beats.load_state_dict(checkpoint["model"])

        spec_aug_config = config.get("spec_aug")
        if spec_aug_config:
            self.beats.add_spec_aug(spec_aug_config)

        self.aggregation_mode = config.get("aggregation_mode", "concat_all")
        self.include_input_embedding = config.get("include_input_embedding", False)
        self.aggregation_hidden_size = config["aggregation_hidden_size"]
        self.encoder_embed_dim = beats_config.encoder_embed_dim

        if self.aggregation_mode == "concat_all":
            num_states = beats_config.encoder_layers
            if self.include_input_embedding:
                num_states += 1

            self.aggregator = ConcatThenCompressAggregator(
                input_dim=self.encoder_embed_dim * num_states,
                output_dim=self.aggregation_hidden_size,
                intermediate_dim=config.get("aggregation_intermediate_dim", 3072),
            )
        elif self.aggregation_mode == "single_layer":
            self.selected_layer_index = config.get("selected_layer_index", -1)
            self.output_projection = self._build_projection_if_needed()
        elif self.aggregation_mode == "weighted_sum":
            num_states = beats_config.encoder_layers
            if self.include_input_embedding:
                num_states += 1

            self.layer_weights = nn.Parameter(torch.ones(num_states, 1))
            self.output_projection = self._build_projection_if_needed()
        else:
            raise ValueError(f"Unsupported BEATs aggregation mode: {self.aggregation_mode}")

    def _build_projection_if_needed(self) -> nn.Module | None:
        if self.encoder_embed_dim == self.aggregation_hidden_size:
            return None

        return nn.Linear(self.encoder_embed_dim, self.aggregation_hidden_size)

    def _select_hidden_states(self, hidden_states: list[torch.Tensor]) -> list[torch.Tensor]:
        # The adapted BEATs implementation returns one extra state before the
        # transformer layers. The 2024 report talks about encoder layers'
        # outputs, so by default we exclude the input embedding and keep only
        # the real transformer layer outputs.
        if self.include_input_embedding:
            return hidden_states

        return hidden_states[1:]

    def _aggregate_hidden_states(self, hidden_states: list[torch.Tensor]) -> torch.Tensor:
        selected_hidden_states = self._select_hidden_states(hidden_states)

        if self.aggregation_mode == "concat_all":
            return self.aggregator(selected_hidden_states)

        if self.aggregation_mode == "single_layer":
            sequence = selected_hidden_states[self.selected_layer_index]
            if self.output_projection is not None:
                sequence = self.output_projection(sequence)
            return sequence

        if self.aggregation_mode == "weighted_sum":
            normalized_weights = nn.functional.softmax(self.layer_weights, dim=0)
            stacked_states = torch.stack(selected_hidden_states, dim=-2)
            sequence = (stacked_states * normalized_weights).sum(dim=-2)
            if self.output_projection is not None:
                sequence = self.output_projection(sequence)
            return sequence

        raise RuntimeError("Unreachable aggregation branch.")

    def forward(
        self,
        waveforms: torch.Tensor,
        waveform_lengths: torch.Tensor,
        sample_rates: torch.Tensor,
    ) -> EncoderOutput:
        if self.sample_rate != 16000:
            raise ValueError(
                "The vendored BEATs preprocessing is defined for 16 kHz input. "
                f"Got sample_rate={self.sample_rate}."
            )

        waveforms, waveform_lengths = resample_batch_waveforms(
            waveforms=waveforms,
            waveform_lengths=waveform_lengths,
            sample_rates=sample_rates,
            target_sample_rate=self.sample_rate,
        )

        padding_mask = torch.arange(
            waveforms.size(1), device=waveforms.device
        ).unsqueeze(0) >= waveform_lengths.unsqueeze(1)

        outputs = self.beats(
            source=waveforms,
            padding_mask=padding_mask,
            max_layer=None,
        )

        aggregated_sequence = self._aggregate_hidden_states(outputs.hidden_states)
        return EncoderOutput(
            sequence=aggregated_sequence,
            padding_mask=outputs.attention_mask,
            hidden_states=outputs.hidden_states,
        )