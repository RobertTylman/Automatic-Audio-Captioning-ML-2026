import torch
import torch.nn as nn
from transformers import ConvNextConfig, ConvNextModel

class ConvNextEncoder(nn.Module):
    def __init__(self, pretrained_model_name_or_path="facebook/convnext-tiny-224", use_pretrained=True):
        super().__init__()
        if use_pretrained:
            self.model = ConvNextModel.from_pretrained(pretrained_model_name_or_path)
        else:
            config = ConvNextConfig()
            self.model = ConvNextModel(config)
            
        self.hidden_size = self.model.config.hidden_sizes[-1]

    def forward(self, fbank):
        """
        fbank: Tensor of shape (batch_size, time_frames, mel_bins) or (batch_size, 1, time_frames, mel_bins)
        """
        if fbank.dim() == 3:
            # (batch, time, mel_bins) -> (batch, 1, time, mel_bins)
            fbank = fbank.unsqueeze(1)
        
        # ConvNeXt expects 3 channels. If we have 1 channel fbank, repeat it 3 times.
        if fbank.size(1) == 1:
            fbank = fbank.expand(-1, 3, -1, -1)
            
        # The transformers ConvNextModel expects (batch_size, num_channels, height, width)
        # So we treat time_frames as height and mel_bins as width
        outputs = self.model(pixel_values=fbank)
        
        # ConvNext outputs last_hidden_state of shape (batch_size, hidden_size, height/32, width/32)
        last_hidden_state = outputs.last_hidden_state
        
        # We need to flatten the spatial dimensions to treat it as a sequence
        batch_size, hidden_size, h, w = last_hidden_state.shape
        
        # Reshape to (batch_size, hidden_size, sequence_length)
        sequence_output = last_hidden_state.view(batch_size, hidden_size, h * w)
        
        # Transpose to (batch_size, sequence_length, hidden_size)
        sequence_output = sequence_output.transpose(1, 2).contiguous()
        
        return sequence_output
