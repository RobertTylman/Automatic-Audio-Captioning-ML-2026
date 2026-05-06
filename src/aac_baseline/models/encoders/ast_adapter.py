import torch
import torch.nn as nn
import torchaudio.compliance.kaldi as ta_kaldi

from ..common import EncoderOutput, resample_batch_waveforms
from .ast_vendor import ASTModel
from .base import AudioEncoderBase


# AudioSet-derived defaults; override per-dataset via the ``ast_encoder``
# config block (e.g. compute Clotho stats once and put them in the YAML).
_AST_AUDIOSET_FBANK_MEAN = -4.2677393
_AST_AUDIOSET_FBANK_STD = 4.5689974


class ASTEncoderAdapter(AudioEncoderBase):
    """Project-facing wrapper around the vendored AST trunk.

    Mirrors the responsibilities of ``BeatsEncoderAdapter``: own the
    preprocessing path (kaldi fbank with AST options + dataset z-norm + pad
    or truncate to ``input_tdim``), call the AST trunk, and return an
    ``EncoderOutput`` that the fusion module knows how to consume.

    The padding mask is all-False because AST hard-pads/truncates each clip
    to a fixed-length window (``input_tdim`` frames) before forward. Trailing
    patches that came from zero-padded silence are still emitted as tokens,
    but treating them as valid is consistent with how AST is normally used
    (a fixed-window classifier). If you need exact length tracking, this is
    where to add it.
    """

    def __init__(self, config: dict, sample_rate: int = 16000) -> None:
        super().__init__()
        self.sample_rate = config.get("target_sample_rate", sample_rate)
        if self.sample_rate != 16000:
            raise ValueError(
                "AST preprocessing assumes 16 kHz input. "
                f"Got target_sample_rate={self.sample_rate}."
            )

        self.input_fdim = config.get("input_fdim", 128)
        self.input_tdim = config.get("input_tdim", 1024)
        self.fbank_mean = config.get("fbank_mean", _AST_AUDIOSET_FBANK_MEAN)
        self.fbank_std = config.get("fbank_std", _AST_AUDIOSET_FBANK_STD)

        self.ast = ASTModel(
            label_dim=config.get("label_dim", 527),
            fstride=config.get("fstride", 10),
            tstride=config.get("tstride", 10),
            input_fdim=self.input_fdim,
            input_tdim=self.input_tdim,
            imagenet_pretrain=config.get("imagenet_pretrain", True),
            audioset_pretrain=config.get("audioset_pretrain", False),
            model_size=config.get("model_size", "base384"),
            verbose=config.get("verbose", False),
            pretrained_dir=config.get("pretrained_dir"),
        )
        self.hidden_size = self.ast.original_embedding_dim

    def _waveforms_to_fbank(self, waveforms: torch.Tensor) -> torch.Tensor:
        """Compute AST-style log-mel fbank with dataset z-normalization.

        Differs from BEATs in three ways: no 2**15 waveform scaling, kaldi
        ``htk_compat=True``/``hanning``/``dither=0``, and
        ``(fbank - mean) / (2 * std)``. Pads (zero) or truncates the time
        axis to ``input_tdim`` so AST's positional embedding lines up.
        """
        fbanks = []
        for waveform in waveforms:
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
        fbank = torch.stack(fbanks, dim=0)

        n_frames = fbank.shape[1]
        if n_frames < self.input_tdim:
            fbank = nn.functional.pad(
                fbank, (0, 0, 0, self.input_tdim - n_frames)
            )
        elif n_frames > self.input_tdim:
            fbank = fbank[:, : self.input_tdim, :]

        fbank = (fbank - self.fbank_mean) / (self.fbank_std * 2)
        return fbank

    def forward(
        self,
        waveforms: torch.Tensor,
        waveform_lengths: torch.Tensor,
        sample_rates: torch.Tensor,
    ) -> EncoderOutput:
        waveforms, _ = resample_batch_waveforms(
            waveforms=waveforms,
            waveform_lengths=waveform_lengths,
            sample_rates=sample_rates,
            target_sample_rate=self.sample_rate,
        )

        fbank = self._waveforms_to_fbank(waveforms).to(waveforms.device)
        sequence = self.ast.extract_features(fbank)

        padding_mask = torch.zeros(
            sequence.size(0), sequence.size(1), dtype=torch.bool, device=sequence.device
        )

        return EncoderOutput(sequence=sequence, padding_mask=padding_mask)
