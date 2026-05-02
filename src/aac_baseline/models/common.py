from dataclasses import dataclass

import torch


@dataclass
class EncoderOutput:
    sequence: torch.Tensor
    padding_mask: torch.Tensor


def lengths_to_padding_mask(lengths: torch.Tensor, max_length: int) -> torch.Tensor:
    positions = torch.arange(max_length, device=lengths.device).unsqueeze(0)
    return positions >= lengths.unsqueeze(1)


def downsample_lengths(lengths: torch.Tensor, stride: int) -> torch.Tensor:
    return ((lengths.float() + stride - 1) / stride).floor().long()
