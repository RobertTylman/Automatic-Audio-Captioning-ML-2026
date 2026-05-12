import argparse
import os
import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(Path("/private/tmp") / "aac_matplotlib"))
os.environ.setdefault("NUMBA_CACHE_DIR", str(Path("/private/tmp") / "aac_numba"))
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import gradio as gr
import torch


SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from aac_baseline.data.collators import AudioCaptioningCollator
from aac_baseline.inference import normalize_caption
from scripts.evaluation.generate_outputs import (
    choose_device,
    load_config,
    load_dotenv,
    load_model,
)


def disable_pretraining_bootstrap(config: dict) -> dict:
    """Avoid loading placeholder upstream weights before the trained checkpoint."""
    model_config = config["model"]

    beats_config = model_config.get("beats_encoder")
    if beats_config is not None:
        beats_config["pretrained_checkpoint_path"] = None

    convnext_config = model_config.get("convnext_encoder")
    if convnext_config is not None:
        convnext_config["pretrained_checkpoint_path"] = None

    ast_config = model_config.get("ast_encoder")
    if ast_config is not None:
        ast_config["pretrained_checkpoint_path"] = None
        ast_config["imagenet_pretrain"] = False
        ast_config["audioset_pretrain"] = False

    return config


class AudioCaptioningDemo:
    def __init__(self, args: argparse.Namespace) -> None:
        load_dotenv(Path(args.env_file).expanduser().resolve())
        config = load_config(Path(args.config).expanduser().resolve())
        if args.skip_pretraining_bootstrap:
            config = disable_pretraining_bootstrap(config)

        self.device = choose_device(args.device)
        self.model = load_model(
            config=config,
            checkpoint_path=Path(args.checkpoint).expanduser().resolve(),
            device=self.device,
            strict=args.strict_checkpoint_load and not args.allow_partial_checkpoint,
        )
        print(f"[demo] model ready on device={self.device}", flush=True)
        self.collator = AudioCaptioningCollator(
            tokenizer_name=config["data"]["tokenizer_name"],
            max_caption_tokens=config["data"]["max_caption_tokens"],
        )
        print("[demo] tokenizer ready", flush=True)
        self.max_length = args.max_length
        self.min_length = args.min_length
        self.no_repeat_ngram_size = args.no_repeat_ngram_size

    def _load_audio(self, audio_input) -> tuple[torch.Tensor, int]:
        if isinstance(audio_input, dict):
            audio_path = audio_input.get("path") or audio_input.get("name")
        elif hasattr(audio_input, "path"):
            audio_path = audio_input.path
        else:
            audio_path = audio_input

        if audio_path is None:
            raise ValueError("Upload an audio file first.")

        import torchaudio

        try:
            waveform, sample_rate = torchaudio.load(audio_path)
        except Exception:
            import librosa

            samples, sample_rate = librosa.load(audio_path, sr=None, mono=True)
            waveform = torch.as_tensor(samples, dtype=torch.float32).unsqueeze(0)

        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        return waveform.squeeze(0), int(sample_rate)

    def caption(self, audio_path: str | dict | None) -> str:
        try:
            waveform, sample_rate = self._load_audio(audio_path)

            display_name = (
                Path(audio_path.get("path", "uploaded_audio")).name
                if isinstance(audio_path, dict)
                else Path(getattr(audio_path, "path", audio_path)).name
            )

            batch = self.collator(
                [
                    {
                        "sample_id": display_name,
                        "audio_path": display_name,
                        "waveform": waveform,
                        "num_samples": int(waveform.numel()),
                        "caption_text": "",
                        "all_captions": [],
                        "sample_rate": int(sample_rate),
                    }
                ]
            )

            with torch.no_grad():
                token_ids = self.model.generate(
                    waveforms=batch["waveforms"].to(self.device),
                    waveform_lengths=batch["waveform_lengths"].to(self.device),
                    sample_rates=batch["sample_rates"].to(self.device),
                    max_length=self.max_length,
                    min_length=self.min_length,
                    do_sample=False,
                    num_return_sequences=1,
                    no_repeat_ngram_size=self.no_repeat_ngram_size,
                )
            caption = self.collator.tokenizer.batch_decode(
                token_ids.detach().cpu(),
                skip_special_tokens=True,
            )[0]
            return normalize_caption(caption)
        except Exception as exc:
            traceback.print_exc()
            return f"Error while captioning audio: {exc}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch a Gradio audio captioning demo.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--env-file", type=str, default=str(REPO_ROOT / ".env"))
    parser.add_argument("--max-length", type=int, default=32)
    parser.add_argument("--min-length", type=int, default=0)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=3)
    parser.add_argument(
        "--allow-partial-checkpoint",
        action="store_true",
        help="Deprecated alias for the default demo behavior.",
    )
    parser.add_argument(
        "--strict-checkpoint-load",
        action="store_true",
        help="Fail if the checkpoint has missing or unexpected keys.",
    )
    parser.add_argument(
        "--skip-pretraining-bootstrap",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Skip loading upstream BEATs/ConvNeXt/AST bootstrap checkpoints from "
            "the YAML. Keep this enabled when loading a full trained checkpoint."
        ),
    )
    parser.add_argument("--server-name", type=str, default="127.0.0.1")
    parser.add_argument(
        "--server-port",
        type=int,
        default=None,
        help="Port for the local Gradio server. If omitted, Gradio picks an open port.",
    )
    parser.add_argument("--share", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    demo_runner = AudioCaptioningDemo(args)
    print("[demo] launching gradio", flush=True)
    interface = gr.Interface(
        fn=demo_runner.caption,
        inputs=gr.Audio(type="filepath", label="Audio file"),
        outputs=gr.Textbox(label="Caption"),
        title="Automatic Audio Captioning",
    )
    interface.launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=args.share,
        show_error=True,
    )


if __name__ == "__main__":
    main()
