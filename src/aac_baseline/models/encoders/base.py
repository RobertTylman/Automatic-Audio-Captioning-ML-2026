import torch.nn as nn


class AudioEncoderBase(nn.Module):
    """
    Shared interface for all audio encoders in this project.

    Each encoder is expected to:
    - receive padded raw waveforms
    - receive waveform lengths
    - return a sequence representation plus a padding mask
    """

    def freeze(self) -> None:
        for parameter in self.parameters():
            parameter.requires_grad = False

    def unfreeze(self) -> None:
        for parameter in self.parameters():
            parameter.requires_grad = True
