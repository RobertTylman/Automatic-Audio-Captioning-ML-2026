import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torchlibrosa.stft import LogmelFilterBank, Spectrogram

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # Simple implementation of truncated normal initialization
    with torch.no_grad():
        return tensor.normal_(mean, std).clamp_(a * std + mean, b * std + mean)

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when used in main path of residual blocks)."""
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        output = x.div(keep_prob) * random_tensor
        return output

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return nn.functional.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class ConvNeXtBlock(nn.Module):
    def __init__(self, dim, drop_path=0.0, layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.scale_layer = (
            Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        input_ = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.scale_layer is not None:
            x = self.scale_layer * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        x = input_ + self.drop_path(x)
        return x

class ConvNeXt(nn.Module):
    def __init__(
        self,
        in_chans=1,
        num_classes=527,
        depths=[3, 3, 9, 3],
        dims=[96, 192, 384, 768],
        drop_path_rate=0.0,
        layer_scale_init_value=1e-6,
        sample_rate=32000,
        window_size=1024,
        hop_size=320,
        mel_bins=224,
        fmin=50,
        fmax=14000,
    ):
        super().__init__()

        self.spectrogram_extractor = Spectrogram(
            n_fft=window_size,
            hop_length=hop_size,
            win_length=window_size,
            window="hann",
            center=True,
            pad_mode="reflect",
            freeze_parameters=True,
        )
        self.logmel_extractor = LogmelFilterBank(
            sr=sample_rate,
            n_fft=window_size,
            n_mels=mel_bins,
            fmin=fmin,
            fmax=fmax,
            ref=1.0,
            amin=1e-10,
            top_db=None,
            freeze_parameters=True,
        )

        self.bn0 = nn.BatchNorm2d(mel_bins)

        self.downsample_layers = nn.ModuleList()
        # Stem layer
        stem = nn.Sequential(
            nn.Conv2d(1, dims[0], kernel_size=(4, 4), stride=(4, 4), padding=(0, 0)),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first"),
        )
        self.downsample_layers.append(stem)
        
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[
                    ConvNeXtBlock(
                        dim=dims[i],
                        drop_path=dp_rates[cur + j],
                        layer_scale_init_value=layer_scale_init_value,
                    )
                    for j in range(depths[i])
                ]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)

    def forward_feature(self, waveforms):
        # 1. Extraction (Internal)
        x = self.spectrogram_extractor(waveforms)
        x = self.logmel_extractor(x)
        
        # 2. Preparation
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        # 3. Backbone Stages
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)

        # 4. Global Temporal Pooling (Keep time axis)
        # x shape: [B, C, T, F]
        x = torch.mean(x, dim=3) # Average over frequency bins [B, C, T]
        x = x.transpose(1, 2)    # [B, T, C]
        
        x = self.norm(x)
        return x

def convnext_tiny(pretrained_path=None, **kwargs):
    model = ConvNeXt(
        depths=[3, 3, 9, 3],
        dims=[96, 192, 384, 768],
        **kwargs
    )
    if pretrained_path:
        checkpoint = torch.load(pretrained_path, map_location="cpu")
        # PANNs checkpoints often have the model state dict in a 'model' key
        state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
        
        # Handle 'gamma' -> 'scale_layer' rename if needed
        new_state_dict = {}
        for k, v in state_dict.items():
            if "gamma" in k:
                new_state_dict[k.replace("gamma", "scale_layer")] = v
            else:
                new_state_dict[k] = v
                
        model.load_state_dict(new_state_dict, strict=False)
    return model
