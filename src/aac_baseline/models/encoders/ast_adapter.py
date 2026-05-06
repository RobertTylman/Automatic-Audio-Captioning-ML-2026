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

    AST itself still consumes a fixed-size spectrogram window, but we also
    compute how many output patch tokens came entirely from real (unpadded)
    audio. That lets the rest of the captioning stack ignore patch embeddings
    whose receptive field extends into zero-padded spectrogram frames.
    """

    def __init__(self, config: dict, sample_rate: int = 16000) -> None:
        super().__init__()
        self.sample_rate = config.get("target_sample_rate", sample_rate)
        if self.sample_rate != 16000:
            raise ValueError(
                "AST preprocessing assumes 16 kHz input. "
                f"Got target_sample_rate={self.sample_rate}."
            )

        # Trunk geometry — these change the patch grid, so AST's positional
        # embedding is sized off them at construction time.
        self.input_fdim = config.get("input_fdim", 128)
        self.input_tdim = config.get("input_tdim", 1024)
        self.patch_size = config.get("patch_size", 16)
        self.tstride = config.get("tstride", 10)

        # Z-normalization stats (recompute these on your dataset for best
        # cross-modality transfer; defaults are AudioSet's).
        self.fbank_mean = config.get("fbank_mean", _AST_AUDIOSET_FBANK_MEAN)
        self.fbank_std = config.get("fbank_std", _AST_AUDIOSET_FBANK_STD)
        self.fbank_std_multiplier = config.get("fbank_std_multiplier", 2.0)

        # Kaldi fbank knobs. Defaults match upstream AST's dataloader; the
        # AST pretrained weights are calibrated to these specific values, so
        # change with caution.
        self.frame_shift_ms = config.get("frame_shift_ms", 10)
        self.frame_length_ms = config.get("frame_length_ms", 25)
        self.kaldi_htk_compat = config.get("kaldi_htk_compat", True)
        self.kaldi_window_type = config.get("kaldi_window_type", "hanning")
        self.kaldi_use_energy = config.get("kaldi_use_energy", False)
        self.kaldi_dither = config.get("kaldi_dither", 0.0)

        self.ast = ASTModel(
            label_dim=config.get("label_dim", 527),
            fstride=config.get("fstride", 10),
            tstride=self.tstride,
            input_fdim=self.input_fdim,
            input_tdim=self.input_tdim,
            imagenet_pretrain=config.get("imagenet_pretrain", True),
            audioset_pretrain=config.get("audioset_pretrain", False),
            model_size=config.get("model_size", "base384"),
            verbose=config.get("verbose", False),
            pretrained_dir=config.get("pretrained_dir"),
        )
        checkpoint_path = config.get("pretrained_checkpoint_path")
        self.checkpoint_load_result = None
        if checkpoint_path:
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            state_dict = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
            if any(key.startswith("module.") for key in state_dict.keys()):
                state_dict = {
                    key.removeprefix("module."): value
                    for key, value in state_dict.items()
                }
            self.checkpoint_load_result = self.ast.load_state_dict(
                state_dict,
                strict=config.get("strict_checkpoint_load", False),
            )
        self.hidden_size = self.ast.original_embedding_dim
        self.freq_patch_count, self.time_patch_count = self.ast.get_shape(
            config.get("fstride", 10),
            self.tstride,
            self.input_fdim,
            self.input_tdim,
        )

    def _waveforms_to_fbank(
        self,
        waveforms: torch.Tensor,
        waveform_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute AST-style log-mel fbank with dataset z-normalization.

        Differs from BEATs in three ways: no 2**15 waveform scaling, kaldi
        AST-canonical options (defaults: ``htk_compat=True``/``hanning``/
        ``dither=0``), and ``(fbank - mean) / (std * fbank_std_multiplier)``.
        Pads (zero) or truncates the time axis to ``input_tdim`` so AST's
        positional embedding lines up.
        """
        fbanks = []
        valid_frame_lengths = []
        for waveform, waveform_length in zip(waveforms, waveform_lengths):
            waveform = waveform[: int(waveform_length.item())].unsqueeze(0)
            fbank = ta_kaldi.fbank(
                waveform,
                htk_compat=self.kaldi_htk_compat,
                sample_frequency=self.sample_rate,
                use_energy=self.kaldi_use_energy,
                window_type=self.kaldi_window_type,
                num_mel_bins=self.input_fdim,
                dither=self.kaldi_dither,
                frame_shift=self.frame_shift_ms,
                frame_length=self.frame_length_ms,
            )
            n_frames = fbank.shape[0]
            valid_frame_lengths.append(min(n_frames, self.input_tdim))
            if n_frames < self.input_tdim:
                fbank = nn.functional.pad(
                    fbank, (0, 0, 0, self.input_tdim - n_frames)
                )
            elif n_frames > self.input_tdim:
                fbank = fbank[: self.input_tdim, :]
            fbanks.append(fbank)
        fbank = torch.stack(fbanks, dim=0)

        fbank = (fbank - self.fbank_mean) / (self.fbank_std * self.fbank_std_multiplier)
        return (
            fbank,
            torch.tensor(valid_frame_lengths, dtype=torch.long, device=waveforms.device),
        )

    def _frame_lengths_to_padding_mask(self, frame_lengths: torch.Tensor) -> torch.Tensor:
        """
        Convert real spectrogram-frame counts into AST token masks.

        A token is marked valid only if its full temporal receptive field lies
        inside the real (unpadded) spectrogram region. This matches the user's
        intent of excluding embeddings that were influenced by padded audio.
        """
        if self.patch_size <= 0:
            raise ValueError("AST patch_size must be positive.")

        valid_time_patches = torch.where(
            frame_lengths >= self.patch_size,
            torch.div(frame_lengths - self.patch_size, self.tstride, rounding_mode="floor") + 1,
            torch.zeros_like(frame_lengths),
        )
        valid_time_patches = valid_time_patches.clamp(min=0, max=self.time_patch_count)

        time_positions = torch.arange(self.time_patch_count, device=frame_lengths.device)
        invalid_time = time_positions.unsqueeze(0) >= valid_time_patches.unsqueeze(1)
        invalid_grid = invalid_time.unsqueeze(1).expand(-1, self.freq_patch_count, -1)
        return invalid_grid.reshape(frame_lengths.size(0), self.freq_patch_count * self.time_patch_count)

    def forward(
        self,
        waveforms: torch.Tensor,
        waveform_lengths: torch.Tensor,
        sample_rates: torch.Tensor,
    ) -> EncoderOutput:
        waveforms, resampled_lengths = resample_batch_waveforms(
            waveforms=waveforms,
            waveform_lengths=waveform_lengths,
            sample_rates=sample_rates,
            target_sample_rate=self.sample_rate,
        )

        fbank, valid_frame_lengths = self._waveforms_to_fbank(waveforms, resampled_lengths)
        fbank = fbank.to(waveforms.device)
        sequence = self.ast.extract_features(fbank)
        padding_mask = self._frame_lengths_to_padding_mask(valid_frame_lengths).to(sequence.device)

        return EncoderOutput(sequence=sequence, padding_mask=padding_mask)