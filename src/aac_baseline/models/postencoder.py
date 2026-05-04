import torch
import torch.nn as nn
from torchaudio.models import Conformer

from .common import EncoderOutput


class ConformerPostEncoder(nn.Module):
    """
    The conformer refines fused audio features before the decoder consumes them.

    The interface stays simple:
    - input: sequence + padding mask
    - output: sequence + padding mask
    """

    def __init__(self, config: dict) -> None:
        super().__init__()
        self.model = Conformer(
            input_dim=config["input_dim"],
            num_heads=config["num_heads"],
            ffn_dim=config["ffn_dim"],
            num_layers=config["num_layers"],
            depthwise_conv_kernel_size=config["depthwise_conv_kernel_size"],
            dropout=config["dropout"],
        )

    def forward(self, encoder_output: EncoderOutput) -> EncoderOutput:
        valid_lengths = (~encoder_output.padding_mask).sum(dim=1)
        refined_sequence, _ = self.model(encoder_output.sequence, valid_lengths)
        return EncoderOutput(
            sequence=refined_sequence,
            padding_mask=encoder_output.padding_mask,
        )
