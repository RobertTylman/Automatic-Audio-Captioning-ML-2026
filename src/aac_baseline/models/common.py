from dataclasses import dataclass

import torch
import torchaudio


@dataclass
class EncoderOutput:
    sequence: torch.Tensor
    padding_mask: torch.Tensor
    hidden_states: list[torch.Tensor] | None = None


def lengths_to_padding_mask(lengths: torch.Tensor, max_length: int) -> torch.Tensor:
    positions = torch.arange(max_length, device=lengths.device).unsqueeze(0)
    return positions >= lengths.unsqueeze(1)


def downsample_lengths(lengths: torch.Tensor, stride: int) -> torch.Tensor:
    return ((lengths.float() + stride - 1) / stride).floor().long()


def resample_batch_waveforms(
    waveforms: torch.Tensor,
    waveform_lengths: torch.Tensor,
    sample_rates: torch.Tensor,
    target_sample_rate: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Resample each waveform in a batch to the encoder's preferred sample rate.

    The dataset intentionally keeps native sample rates so different encoders
    can own their own preprocessing. This helper converts a mixed-sample-rate
    batch into one resampled, padded batch for a specific encoder branch.
    """

    resampled_waveforms = []
    resampled_lengths = []

    for index in range(waveforms.size(0)):
        waveform = waveforms[index, : waveform_lengths[index]].unsqueeze(0)
        original_sample_rate = int(sample_rates[index].item())

        if original_sample_rate != target_sample_rate:
            waveform = torchaudio.functional.resample(
                waveform,
                orig_freq=original_sample_rate,
                new_freq=target_sample_rate,
            )

        waveform = waveform.squeeze(0)
        resampled_waveforms.append(waveform)
        resampled_lengths.append(waveform.numel())

    max_length = max(resampled_lengths)
    padded_waveforms = [
        torch.nn.functional.pad(waveform, (0, max_length - waveform.numel()))
        for waveform in resampled_waveforms
    ]

    return (
        torch.stack(padded_waveforms, dim=0),
        torch.tensor(resampled_lengths, dtype=torch.long, device=waveforms.device),
    )