# -*- coding: utf-8 -*-
# Adapted from https://github.com/YuanGongND/ast/blob/master/src/models/ast_models.py
# Original: Yuan Gong, MIT (yuangong@mit.edu)
#
# Refactored to match this project's encoder conventions (see BEATs.py, ConvNext.py):
#   - no os.environ side effects at import time
#   - pretrained download dir is configurable via constructor
#   - uses torch.hub for downloads (no `wget` dependency)
#   - exposes an ASTEncoder wrapper that returns sequence features [B, T, D]
#     for fusion, mirroring the ConvNextEncoder interface.

import os
import warnings
from typing import Optional

import torch
import torch.nn as nn
import torchaudio.compliance.kaldi as ta_kaldi

import timm
from timm.models.layers import to_2tuple, trunc_normal_


# Default fbank stats are from AudioSet (the original AST training set).
# Override for other datasets — see compute_dataset_stats() in upstream AST repo.
_AST_AUDIOSET_FBANK_MEAN = -4.2677393
_AST_AUDIOSET_FBANK_STD = 4.5689974


_AUDIOSET_CKPT_URL = "https://www.dropbox.com/s/cv4knew8mvbrnvq/audioset_0.4593.pth?dl=1"
_AUDIOSET_CKPT_NAME = "audioset_10_10_0.4593.pth"


def _default_pretrained_dir() -> str:
    return os.environ.get(
        "AST_PRETRAINED_DIR",
        os.path.join(os.path.expanduser("~"), ".cache", "torch", "ast"),
    )


# override the timm package to relax the input shape constraint.
class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class ASTModel(nn.Module):
    """
    The AST model.

    :param label_dim: number of total classes (527 for AudioSet, 50 for ESC-50, 35 for SC v2-35)
    :param fstride: stride of patch splitting on the frequency dim
    :param tstride: stride of patch splitting on the time dim
    :param input_fdim: number of frequency bins of the input spectrogram
    :param input_tdim: number of time frames of the input spectrogram
    :param imagenet_pretrain: whether to use ImageNet pretrained model
    :param audioset_pretrain: whether to use ImageNet + AudioSet pretrained model
    :param model_size: one of [tiny224, small224, base224, base384]
    :param pretrained_dir: directory in which to look for / download the AudioSet ckpt
        and where timm will cache ImageNet weights. Defaults to $AST_PRETRAINED_DIR or
        ~/.cache/torch/ast. Replaces the upstream relative-path '../../pretrained_models'.
    """

    def __init__(
        self,
        label_dim: int = 527,
        fstride: int = 10,
        tstride: int = 10,
        input_fdim: int = 128,
        input_tdim: int = 1024,
        imagenet_pretrain: bool = True,
        audioset_pretrain: bool = False,
        model_size: str = "base384",
        verbose: bool = True,
        pretrained_dir: Optional[str] = None,
    ):
        super().__init__()

        if timm.__version__ != "0.4.5":
            warnings.warn(
                f"AST was developed against timm==0.4.5; running with timm=={timm.__version__}. "
                "If you hit attribute or shape errors, pin to 0.4.5.",
                RuntimeWarning,
            )

        # Resolve pretrained dir and point timm at it ONLY if a dir was supplied
        # (or the AST_PRETRAINED_DIR env var is set). We never overwrite an
        # existing TORCH_HOME the caller has set.
        self.pretrained_dir = pretrained_dir or _default_pretrained_dir()
        os.makedirs(self.pretrained_dir, exist_ok=True)
        os.environ.setdefault("TORCH_HOME", self.pretrained_dir)

        if verbose:
            print("---------------AST Model Summary---------------")
            print(
                "ImageNet pretraining: {}, AudioSet pretraining: {}".format(
                    imagenet_pretrain, audioset_pretrain
                )
            )

        # override timm input shape restriction
        timm.models.vision_transformer.PatchEmbed = PatchEmbed

        if not audioset_pretrain:
            if model_size == "tiny224":
                self.v = timm.create_model("vit_deit_tiny_distilled_patch16_224", pretrained=imagenet_pretrain)
            elif model_size == "small224":
                self.v = timm.create_model("vit_deit_small_distilled_patch16_224", pretrained=imagenet_pretrain)
            elif model_size == "base224":
                self.v = timm.create_model("vit_deit_base_distilled_patch16_224", pretrained=imagenet_pretrain)
            elif model_size == "base384":
                self.v = timm.create_model("vit_deit_base_distilled_patch16_384", pretrained=imagenet_pretrain)
            else:
                raise ValueError("Model size must be one of tiny224, small224, base224, base384.")
            self.original_num_patches = self.v.patch_embed.num_patches
            self.oringal_hw = int(self.original_num_patches ** 0.5)
            self.original_embedding_dim = self.v.pos_embed.shape[2]
            self.mlp_head = nn.Sequential(
                nn.LayerNorm(self.original_embedding_dim),
                nn.Linear(self.original_embedding_dim, label_dim),
            )

            f_dim, t_dim = self.get_shape(fstride, tstride, input_fdim, input_tdim)
            num_patches = f_dim * t_dim
            self.v.patch_embed.num_patches = num_patches
            if verbose:
                print("frequency stride={:d}, time stride={:d}".format(fstride, tstride))
                print("number of patches={:d}".format(num_patches))

            new_proj = nn.Conv2d(1, self.original_embedding_dim, kernel_size=(16, 16), stride=(fstride, tstride))
            if imagenet_pretrain:
                new_proj.weight = nn.Parameter(torch.sum(self.v.patch_embed.proj.weight, dim=1).unsqueeze(1))
                new_proj.bias = self.v.patch_embed.proj.bias
            self.v.patch_embed.proj = new_proj

            if imagenet_pretrain:
                new_pos_embed = (
                    self.v.pos_embed[:, 2:, :]
                    .detach()
                    .reshape(1, self.original_num_patches, self.original_embedding_dim)
                    .transpose(1, 2)
                    .reshape(1, self.original_embedding_dim, self.oringal_hw, self.oringal_hw)
                )
                if t_dim <= self.oringal_hw:
                    new_pos_embed = new_pos_embed[
                        :,
                        :,
                        :,
                        int(self.oringal_hw / 2) - int(t_dim / 2) : int(self.oringal_hw / 2) - int(t_dim / 2) + t_dim,
                    ]
                else:
                    new_pos_embed = nn.functional.interpolate(new_pos_embed, size=(self.oringal_hw, t_dim), mode="bilinear")
                if f_dim <= self.oringal_hw:
                    new_pos_embed = new_pos_embed[
                        :,
                        :,
                        int(self.oringal_hw / 2) - int(f_dim / 2) : int(self.oringal_hw / 2) - int(f_dim / 2) + f_dim,
                        :,
                    ]
                else:
                    new_pos_embed = nn.functional.interpolate(new_pos_embed, size=(f_dim, t_dim), mode="bilinear")
                new_pos_embed = new_pos_embed.reshape(1, self.original_embedding_dim, num_patches).transpose(1, 2)
                self.v.pos_embed = nn.Parameter(torch.cat([self.v.pos_embed[:, :2, :].detach(), new_pos_embed], dim=1))
            else:
                new_pos_embed = nn.Parameter(
                    torch.zeros(1, self.v.patch_embed.num_patches + 2, self.original_embedding_dim)
                )
                self.v.pos_embed = new_pos_embed
                trunc_normal_(self.v.pos_embed, std=0.02)

        else:
            if not imagenet_pretrain:
                raise ValueError(
                    "currently model pretrained on only audioset is not supported, "
                    "please set imagenet_pretrain = True to use audioset pretrained model."
                )
            if model_size != "base384":
                raise ValueError("currently only has base384 AudioSet pretrained model.")

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            ckpt_path = os.path.join(self.pretrained_dir, _AUDIOSET_CKPT_NAME)
            if not os.path.exists(ckpt_path):
                if verbose:
                    print(f"[AST] downloading AudioSet checkpoint to {ckpt_path}")
                torch.hub.download_url_to_file(_AUDIOSET_CKPT_URL, ckpt_path, progress=verbose)

            sd = torch.load(ckpt_path, map_location=device)
            audio_model = ASTModel(
                label_dim=527,
                fstride=10,
                tstride=10,
                input_fdim=128,
                input_tdim=1024,
                imagenet_pretrain=False,
                audioset_pretrain=False,
                model_size="base384",
                verbose=False,
                pretrained_dir=self.pretrained_dir,
            )
            audio_model = nn.DataParallel(audio_model)
            audio_model.load_state_dict(sd, strict=False)
            self.v = audio_model.module.v
            self.original_embedding_dim = self.v.pos_embed.shape[2]
            self.mlp_head = nn.Sequential(
                nn.LayerNorm(self.original_embedding_dim),
                nn.Linear(self.original_embedding_dim, label_dim),
            )

            f_dim, t_dim = self.get_shape(fstride, tstride, input_fdim, input_tdim)
            num_patches = f_dim * t_dim
            self.v.patch_embed.num_patches = num_patches
            if verbose:
                print("frequency stride={:d}, time stride={:d}".format(fstride, tstride))
                print("number of patches={:d}".format(num_patches))

            new_pos_embed = (
                self.v.pos_embed[:, 2:, :]
                .detach()
                .reshape(1, 1212, 768)
                .transpose(1, 2)
                .reshape(1, 768, 12, 101)
            )
            if t_dim < 101:
                new_pos_embed = new_pos_embed[:, :, :, 50 - int(t_dim / 2) : 50 - int(t_dim / 2) + t_dim]
            else:
                new_pos_embed = nn.functional.interpolate(new_pos_embed, size=(12, t_dim), mode="bilinear")
            if f_dim < 12:
                new_pos_embed = new_pos_embed[:, :, 6 - int(f_dim / 2) : 6 - int(f_dim / 2) + f_dim, :]
            elif f_dim > 12:
                new_pos_embed = nn.functional.interpolate(new_pos_embed, size=(f_dim, t_dim), mode="bilinear")
            new_pos_embed = new_pos_embed.reshape(1, 768, num_patches).transpose(1, 2)
            self.v.pos_embed = nn.Parameter(torch.cat([self.v.pos_embed[:, :2, :].detach(), new_pos_embed], dim=1))

    def get_shape(self, fstride, tstride, input_fdim=128, input_tdim=1024):
        test_input = torch.randn(1, 1, input_fdim, input_tdim)
        test_proj = nn.Conv2d(1, self.original_embedding_dim, kernel_size=(16, 16), stride=(fstride, tstride))
        test_out = test_proj(test_input)
        f_dim = test_out.shape[2]
        t_dim = test_out.shape[3]
        return f_dim, t_dim

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Run the transformer trunk and return per-patch features.

        :param x: spectrogram of shape (B, T, F) -- e.g., (B, 1024, 128)
        :return: features of shape (B, num_patches, embed_dim). cls/dist tokens are stripped.
        """
        x = x.unsqueeze(1)
        x = x.transpose(2, 3)

        B = x.shape[0]
        x = self.v.patch_embed(x)
        cls_tokens = self.v.cls_token.expand(B, -1, -1)
        dist_token = self.v.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)
        x = x + self.v.pos_embed
        x = self.v.pos_drop(x)
        for blk in self.v.blocks:
            x = blk(x)
        x = self.v.norm(x)
        return x[:, 2:, :]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: spectrogram of shape (B, T, F), e.g., (B, 1024, 128)
        :return: classification logits of shape (B, label_dim)
        """
        x = x.unsqueeze(1)
        x = x.transpose(2, 3)

        B = x.shape[0]
        x = self.v.patch_embed(x)
        cls_tokens = self.v.cls_token.expand(B, -1, -1)
        dist_token = self.v.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)
        x = x + self.v.pos_embed
        x = self.v.pos_drop(x)
        for blk in self.v.blocks:
            x = blk(x)
        x = self.v.norm(x)
        x = (x[:, 0] + x[:, 1]) / 2

        return self.mlp_head(x)


class ASTEncoder(nn.Module):
    """Sequence-feature wrapper around ASTModel for use in the fusion pipeline.

    Mirrors ConvNextEncoder: takes an fbank tensor and returns a sequence
    representation [B, T, D] suitable for concatenation with BEATs features.
    """

    def __init__(
        self,
        input_fdim: int = 128,
        input_tdim: int = 1024,
        fstride: int = 10,
        tstride: int = 10,
        model_size: str = "base384",
        imagenet_pretrain: bool = True,
        audioset_pretrain: bool = False,
        pretrained_dir: Optional[str] = None,
        verbose: bool = False,
        fbank_mean: float = _AST_AUDIOSET_FBANK_MEAN,
        fbank_std: float = _AST_AUDIOSET_FBANK_STD,
    ):
        super().__init__()
        self.model = ASTModel(
            label_dim=527,
            fstride=fstride,
            tstride=tstride,
            input_fdim=input_fdim,
            input_tdim=input_tdim,
            imagenet_pretrain=imagenet_pretrain,
            audioset_pretrain=audioset_pretrain,
            model_size=model_size,
            verbose=verbose,
            pretrained_dir=pretrained_dir,
        )
        self.hidden_size = self.model.original_embedding_dim
        self.input_tdim = input_tdim
        self.input_fdim = input_fdim
        self.fbank_mean = fbank_mean
        self.fbank_std = fbank_std

    def preprocess(self, source: torch.Tensor) -> torch.Tensor:
        """Compute AST-style log-mel fbank with dataset z-normalization.

        Differs from BEATs.preprocess in three ways:
        1. No 2**15 waveform scaling (works on the [-1, 1] float32 waveform).
        2. Kaldi options: htk_compat=True, hanning window, dither=0, use_energy=False.
        3. Z-normalization is `(fbank - mean) / (std * 2)` using AST's stats
           (default = AudioSet's), not BEATs's stats.

        The fbank is then padded (zero) or center-truncated to exactly
        ``input_tdim`` frames so that AST's positional embedding (sized at
        construction time) lines up with the patch grid. This mirrors
        upstream AST's dataloader behavior.

        :param source: waveform tensor of shape (B, samples), values in [-1, 1].
        :return: fbank of shape (B, input_tdim, num_mel_bins).
        """
        fbanks = []
        for waveform in source:
            waveform = waveform.unsqueeze(0)
            fbank = ta_kaldi.fbank(
                waveform,
                htk_compat=True,
                sample_frequency=16000,
                use_energy=False,
                window_type="hanning",
                num_mel_bins=self.input_fdim,
                dither=0.0,
                frame_shift=10,
            )
            fbanks.append(fbank)
        fbank = torch.stack(fbanks, dim=0)  # (B, T, F)

        # Pad (zero) or truncate to input_tdim frames.
        # F.pad expects (last_dim_left, last_dim_right, second_last_left, second_last_right).
        n_frames = fbank.shape[1]
        if n_frames < self.input_tdim:
            fbank = torch.nn.functional.pad(
                fbank, (0, 0, 0, self.input_tdim - n_frames)
            )
        elif n_frames > self.input_tdim:
            fbank = fbank[:, : self.input_tdim, :]

        fbank = (fbank - self.fbank_mean) / (self.fbank_std * 2)
        return fbank

    def forward(self, fbank: torch.Tensor) -> torch.Tensor:
        """
        :param fbank: spectrogram tensor. Accepts (B, T, F) or (B, 1, T, F).
            Should be the output of :meth:`preprocess` for best results.
        :return: sequence features (B, num_patches, hidden_size).
        """
        if fbank.dim() == 4:
            if fbank.size(1) != 1:
                raise ValueError(f"expected single-channel fbank, got {fbank.size(1)} channels")
            fbank = fbank.squeeze(1)
        return self.model.extract_features(fbank)


if __name__ == "__main__":
    input_tdim = 100
    ast_mdl = ASTModel(input_tdim=input_tdim)
    test_input = torch.rand([10, input_tdim, 128])
    test_output = ast_mdl(test_input)
    print(test_output.shape)
