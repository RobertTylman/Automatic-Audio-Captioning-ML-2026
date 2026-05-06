import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchlibrosa.augmentation import SpecAugmentation
from torchlibrosa.stft import LogmelFilterBank, Spectrogram


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    with torch.no_grad():
        return tensor.normal_(mean, std).clamp_(a * std + mean, b * std + mean)


def drop_path(
    x: Tensor,
    drop_prob: float = 0.0,
    training: bool = False,
    scale_by_keep: bool = True,
) -> Tensor:
    if drop_prob == 0.0 or not training:
        return x

    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(
                x,
                self.normalized_shape,
                self.weight,
                self.bias,
                self.eps,
            )

        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class Block(nn.Module):
    def __init__(self, dim, drop_path=0.0, layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        residual = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)
        return residual + self.drop_path(x)


def build_audio_stem(stem_out_channels: int, after_stem_dim: tuple[int, ...]) -> tuple[nn.Conv2d, dict]:
    """
    Mirror the source repo's audio stem replacement logic so the checkpoint sees
    the exact first convolution geometry it was trained with.
    """

    dims = tuple(after_stem_dim)

    if len(dims) < 2:
        if dims[0] == 56:
            conv = nn.Conv2d(
                1,
                stem_out_channels,
                kernel_size=(18, 4),
                stride=(18, 4),
                padding=(9, 0),
            )
            spec = {"kernel_size": 18, "stride": 18, "padding": 9}
        elif dims[0] == 112:
            conv = nn.Conv2d(
                1,
                stem_out_channels,
                kernel_size=(9, 2),
                stride=(9, 2),
                padding=(4, 0),
            )
            spec = {"kernel_size": 9, "stride": 9, "padding": 4}
        else:
            raise ValueError("after_stem_dim must be 56, 112, or [252, 56] variants")
    else:
        if dims == (252, 56):
            conv = nn.Conv2d(
                1,
                stem_out_channels,
                kernel_size=(4, 4),
                stride=(4, 4),
                padding=(4, 0),
            )
            spec = {"kernel_size": 4, "stride": 4, "padding": 4}
        elif dims == (504, 28):
            conv = nn.Conv2d(
                1,
                stem_out_channels,
                kernel_size=(4, 8),
                stride=(2, 8),
                padding=(5, 0),
            )
            spec = {"kernel_size": 4, "stride": 2, "padding": 5}
        elif dims == (504, 56):
            conv = nn.Conv2d(
                1,
                stem_out_channels,
                kernel_size=(4, 4),
                stride=(2, 4),
                padding=(5, 0),
            )
            spec = {"kernel_size": 4, "stride": 2, "padding": 5}
        else:
            raise ValueError("after_stem_dim must be 56, 112, or [252, 56] variants")

    trunc_normal_(conv.weight, std=0.02)
    nn.init.constant_(conv.bias, 0)
    return conv, spec


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
        enable_spec_augment=False,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.hop_size = hop_size
        self.enable_spec_augment = enable_spec_augment

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

        self.spec_augmenter = SpecAugmentation(
            time_drop_width=64,
            time_stripes_num=2,
            freq_drop_width=28,
            freq_stripes_num=2,
        )

        self.bn0 = nn.BatchNorm2d(mel_bins)

        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(3, dims[0], kernel_size=(4, 4), stride=(4, 4)),
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
                    Block(
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
        self.head_audioset = nn.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            trunc_normal_(module.weight, std=0.02)
            nn.init.constant_(module.bias, 0)

    def _extract_logmel(self, waveforms: torch.Tensor) -> torch.Tensor:
        x = self.spectrogram_extractor(waveforms)
        x = self.logmel_extractor(x)
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)
        if self.training and self.enable_spec_augment:
            x = self.spec_augmenter(x)
        return x

    def forward_features(self, x, return_frame_embeddings: bool = False):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)

        if return_frame_embeddings:
            return x

        x = torch.mean(x, dim=3)
        x_max, _ = torch.max(x, dim=2)
        x_mean = torch.mean(x, dim=2)
        return self.norm(x_max + x_mean)

    def forward(self, waveforms):
        x = self._extract_logmel(waveforms)
        x = self.forward_features(x)
        logits = self.head_audioset(x)
        return {
            "clipwise_output": torch.sigmoid(logits),
            "clipwise_logits": logits,
        }

    def forward_frame_embeddings(self, waveforms):
        x = self._extract_logmel(waveforms)
        return self.forward_features(x, return_frame_embeddings=True)


def convnext_tiny(
    pretrained_path=None,
    strict=False,
    drop_path_rate=0.0,
    after_stem_dim=(252, 56),
    **kwargs,
):
    model = ConvNeXt(
        in_chans=1,
        num_classes=527,
        depths=[3, 3, 9, 3],
        dims=[96, 192, 384, 768],
        drop_path_rate=drop_path_rate,
        **kwargs,
    )

    stem, stem_spec = build_audio_stem(96, tuple(after_stem_dim))
    model.downsample_layers[0][0] = stem
    model.audio_stem_spec = stem_spec

    if pretrained_path:
        checkpoint = torch.load(pretrained_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
        result = model.load_state_dict(state_dict, strict=strict)
        model.checkpoint_load_result = result

    return model
