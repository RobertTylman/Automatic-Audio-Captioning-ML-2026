import torch
import torch.nn as nn

from ..common import (
    EncoderOutput,
    downsample_lengths,
    lengths_to_padding_mask,
    resample_batch_waveforms,
)
from .base import AudioEncoderBase
from .panns_convnext import convnext_tiny


class ConvNextEncoderAdapter(AudioEncoderBase):
    """
    Adapter for the real audio-native ConvNeXt encoder (PANNs-style).
    Uses the 465mAP checkpoint for state-of-the-art audio feature extraction.
    """

    def __init__(self, config: dict, sample_rate: int = 32000) -> None:
        super().__init__()
        # PANNs ConvNeXt is trained on 32kHz audio.
        self.sample_rate = config.get("target_sample_rate", 32000)
        self.hidden_size = config["hidden_size"]  # Hidden size for ConvNeXt-Tiny

        # 1. Initialize the real architecture blueprint
        self.model = convnext_tiny()

        # 2. Load the specific audio-native checkpoint
        checkpoint_path = config.get("pretrained_checkpoint_path")
        if checkpoint_path:
            print(f"Loading real ConvNeXt weights from: {checkpoint_path}")
            # We load with weights_only=False because the checkpoint contains numpy scalars
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
            
            # Fix gamma -> scale_layer mismatch if necessary
            fixed_state_dict = {}
            for k, v in state_dict.items():
                fixed_state_dict[k.replace("gamma", "scale_layer")] = v
                
            self.model.load_state_dict(fixed_state_dict, strict=False)

    def forward(
        self,
        waveforms: torch.Tensor,
        waveform_lengths: torch.Tensor,
        sample_rates: torch.Tensor,
    ) -> EncoderOutput:
        # Use the project's standard resampling utility
        waveforms, waveform_lengths = resample_batch_waveforms(
            waveforms=waveforms,
            waveform_lengths=waveform_lengths,
            sample_rates=sample_rates,
            target_sample_rate=self.sample_rate,
        )

        # The real model handles STFT and Log-Mel conversion internally.
        # This ensures the features exactly match what the model was trained on.
        features = self.model.forward_feature(waveforms)

        # Note: These calculations safely assume 32kHz audio because `resample_batch_waveforms` 
        # automatically resamples any input and updates `waveform_lengths` above.
        # 1. STFT frames with center=True and hop_size=320 (10ms at 32kHz)
        frame_lengths = (waveform_lengths // 320) + 1
        
        # 2. ConvNeXt architecture downsamples spatial dimensions by 32x.
        frame_lengths = frame_lengths // 32
        
        padding_mask = lengths_to_padding_mask(frame_lengths, features.size(1))

        return EncoderOutput(sequence=features, padding_mask=padding_mask)