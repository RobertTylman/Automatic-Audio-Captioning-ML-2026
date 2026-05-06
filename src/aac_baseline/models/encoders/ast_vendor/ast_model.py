# --------------------------------------------------------
# Audio Spectrogram Transformer (https://arxiv.org/abs/2104.01778)
# Adapted from https://github.com/YuanGongND/ast/blob/master/src/models/ast_models.py
# Original: Yuan Gong, MIT (yuangong@mit.edu)
#
# Vendored copy for the aac_baseline package. The project-facing wrapper lives
# in ``ast_adapter.py``; this module only owns the AST trunk and its
# ImageNet/AudioSet pretrained-weight loading.
# --------------------------------------------------------

import os
import warnings
from typing import Optional

import torch
import torch.nn as nn

import timm
from timm.models.layers import to_2tuple, trunc_normal_


_AUDIOSET_CKPT_URL = "https://www.dropbox.com/s/cv4knew8mvbrnvq/audioset_0.4593.pth?dl=1"
_AUDIOSET_CKPT_NAME = "audioset_10_10_0.4593.pth"


def _default_pretrained_dir() -> str:
    return os.environ.get(
        "AST_PRETRAINED_DIR",
        os.path.join(os.path.expanduser("~"), ".cache", "torch", "ast"),
    )


class PatchEmbed(nn.Module):
    """Replaces timm's PatchEmbed to drop the hardcoded input-shape assertion.

    timm's original PatchEmbed asserts that the input H/W matches a baked-in
    img_size (224 or 384), which is wrong for spectrograms. This drop-in keeps
    the same conv projection but does not check the input shape.
    """

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
    """The AST model: DeiT trunk adapted to consume spectrograms.

    Two surgeries on top of timm's pretrained DeiT:

    1. Patch-embed channel adaptation. DeiT's first conv expects 3 channels
       (RGB); spectrograms are 1-channel. The adapted conv has shape
       ``(embed_dim, 1, 16, 16)`` whose weights are the channel-summed RGB
       weights (equivalent to feeding a gray-replicated RGB image).
    2. Position-embedding interpolation. DeiT-base/384 has a 24x24 patch grid
       (576 positions); AST has a (f, t)-shaped grid that depends on
       ``input_fdim`` / ``input_tdim`` / strides. We reshape DeiT's positions
       into a 24x24 image and bilinear-interpolate / center-crop into the
       target shape.

    Optionally also loads AudioSet-finetuned weights on top, recursively
    constructing a fresh AST and stealing its trunk.
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

        # Override timm's PatchEmbed before constructing the ViT.
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
                    new_pos_embed = nn.functional.interpolate(
                        new_pos_embed, size=(self.oringal_hw, t_dim), mode="bilinear"
                    )
                if f_dim <= self.oringal_hw:
                    new_pos_embed = new_pos_embed[
                        :,
                        :,
                        int(self.oringal_hw / 2) - int(f_dim / 2) : int(self.oringal_hw / 2) - int(f_dim / 2) + f_dim,
                        :,
                    ]
                else:
                    new_pos_embed = nn.functional.interpolate(
                        new_pos_embed, size=(f_dim, t_dim), mode="bilinear"
                    )
                new_pos_embed = new_pos_embed.reshape(
                    1, self.original_embedding_dim, num_patches
                ).transpose(1, 2)
                self.v.pos_embed = nn.Parameter(
                    torch.cat([self.v.pos_embed[:, :2, :].detach(), new_pos_embed], dim=1)
                )
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
                new_pos_embed = new_pos_embed[
                    :, :, :, 50 - int(t_dim / 2) : 50 - int(t_dim / 2) + t_dim
                ]
            else:
                new_pos_embed = nn.functional.interpolate(new_pos_embed, size=(12, t_dim), mode="bilinear")
            if f_dim < 12:
                new_pos_embed = new_pos_embed[
                    :, :, 6 - int(f_dim / 2) : 6 - int(f_dim / 2) + f_dim, :
                ]
            elif f_dim > 12:
                new_pos_embed = nn.functional.interpolate(new_pos_embed, size=(f_dim, t_dim), mode="bilinear")
            new_pos_embed = new_pos_embed.reshape(1, 768, num_patches).transpose(1, 2)
            self.v.pos_embed = nn.Parameter(
                torch.cat([self.v.pos_embed[:, :2, :].detach(), new_pos_embed], dim=1)
            )

    def get_shape(self, fstride, tstride, input_fdim=128, input_tdim=1024):
        test_input = torch.randn(1, 1, input_fdim, input_tdim)
        test_proj = nn.Conv2d(
            1, self.original_embedding_dim, kernel_size=(16, 16), stride=(fstride, tstride)
        )
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
