import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader, Dataset


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from aac_baseline.data.collators import AudioCaptioningCollator
from aac_baseline.data.datasets import ClothoCaptionDataset
from aac_baseline.inference import (
    CaptionRerankingPipeline,
    ClapSimilarityScorer,
    normalize_caption,
)
from aac_baseline.models.captioner import DCASE24BaselineCaptioner


class AudioFileDataset(Dataset):
    def __init__(self, audio_dir: str, file_names: list[str] | None = None) -> None:
        import torchaudio

        self.torchaudio = torchaudio
        self.audio_dir = Path(audio_dir)
        if file_names is None:
            file_names = sorted(path.name for path in self.audio_dir.glob("*.wav"))
        self.examples = [
            {
                "file_name": file_name,
                "audio_path": str(self.audio_dir / file_name),
            }
            for file_name in file_names
        ]
        if not self.examples:
            raise ValueError(f"No audio files found in {self.audio_dir}")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        example = self.examples[index]
        waveform, sample_rate = self.torchaudio.load(example["audio_path"])
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        waveform = waveform.squeeze(0)
        return {
            "sample_id": example["file_name"],
            "audio_path": example["audio_path"],
            "waveform": waveform,
            "num_samples": int(waveform.numel()),
            "caption_text": "",
            "all_captions": [],
            "sample_rate": int(sample_rate),
        }


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_dotenv(env_path: Path) -> None:
    if not env_path.exists():
        return

    with env_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip("\"'"))

    if "OPENAI_API_KEY" not in os.environ and "OPENAI" in os.environ:
        os.environ["OPENAI_API_KEY"] = os.environ["OPENAI"]


def load_checkpoint(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_model(
    config: dict,
    checkpoint_path: Path,
    device: torch.device,
    strict: bool,
) -> DCASE24BaselineCaptioner:
    model = DCASE24BaselineCaptioner(config["model"])
    checkpoint = load_checkpoint(checkpoint_path)
    state_dict = checkpoint.get("state_dict", checkpoint)
    model_state = {}
    for key, value in state_dict.items():
        model_state[key.removeprefix("model.") if key.startswith("model.") else key] = value

    missing, unexpected = model.load_state_dict(model_state, strict=strict)
    print(
        f"[model] loaded checkpoint={checkpoint_path} "
        f"missing_keys={len(missing)} unexpected_keys={len(unexpected)}",
        flush=True,
    )
    if missing:
        print(f"[model] first missing keys: {missing[:8]}", flush=True)
    if unexpected:
        print(f"[model] first unexpected keys: {unexpected[:8]}", flush=True)

    model.to(device)
    model.eval()
    return model


def _first_existing(data_config: dict, keys: list[str]) -> str | None:
    for key in keys:
        value = data_config.get(key)
        if value:
            return value
    return None


def resolve_dataset_paths(
    config: dict,
    split: str,
    audio_dir: str | None,
    caption_csv: str | None,
) -> tuple[str, str | None]:
    if audio_dir is not None:
        return audio_dir, caption_csv

    data_config = config["data"]
    aliases = {
        "train": ["train"],
        "val": ["val", "validation"],
        "validation": ["validation", "val"],
        "eval": ["eval", "evaluation", "test"],
        "evaluation": ["evaluation", "eval", "test"],
        "analysis": ["analysis"],
    }
    prefixes = aliases.get(split, [split])
    audio_keys = [f"{prefix}_audio_dir" for prefix in prefixes]
    caption_keys = [f"{prefix}_caption_csv" for prefix in prefixes]
    resolved_audio = _first_existing(data_config, audio_keys)
    resolved_caption = caption_csv or _first_existing(data_config, caption_keys)
    if resolved_audio is None:
        raise KeyError(
            "Could not resolve an audio directory from the config. "
            "Pass --audio-dir explicitly or add an <split>_audio_dir key."
        )
    return resolved_audio, resolved_caption


def build_dataset(audio_dir: str, caption_csv: str | None) -> Dataset:
    if caption_csv is None:
        return AudioFileDataset(audio_dir=audio_dir)

    dataframe = pd.read_csv(caption_csv, nrows=1)
    if "caption_1" in dataframe.columns:
        return ClothoCaptionDataset(
            audio_dir=audio_dir,
            caption_csv=caption_csv,
            caption_mode="first",
        )

    if "file_name" not in dataframe.columns:
        raise ValueError(f"{caption_csv} must contain a file_name column")
    file_names = pd.read_csv(caption_csv)["file_name"].tolist()
    return AudioFileDataset(audio_dir=audio_dir, file_names=file_names)


def build_dataloader(
    config: dict,
    dataset: Dataset,
    batch_size: int,
    num_workers: int | None,
) -> DataLoader:
    data_config = config["data"]
    collator = AudioCaptioningCollator(
        tokenizer_name=data_config["tokenizer_name"],
        max_caption_tokens=data_config["max_caption_tokens"],
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=data_config.get("num_workers", 0) if num_workers is None else num_workers,
        collate_fn=collator,
        pin_memory=True,
    )


def load_completed(output_csv: Path) -> set[str]:
    if not output_csv.exists():
        return set()
    dataframe = pd.read_csv(output_csv)
    if "file_name" not in dataframe.columns:
        raise ValueError(f"{output_csv} exists but has no file_name column")
    completed = set(dataframe["file_name"].dropna().astype(str))
    print(f"[resume] found {len(completed)} completed rows in {output_csv}", flush=True)
    return completed


def append_prediction(output_csv: Path, file_name: str, caption: str) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    write_header = not output_csv.exists()
    with output_csv.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["file_name", "caption_predicted"])
        if write_header:
            writer.writeheader()
        writer.writerow({"file_name": file_name, "caption_predicted": caption})
        handle.flush()


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()


def load_generation_config(args: argparse.Namespace) -> dict[str, Any]:
    generation = {
        "max_length": args.max_length,
        "min_length": args.min_length,
        "do_sample": args.mode == "rerank",
        "temperature": args.temperature,
        "top_p": args.top_p,
        "num_return_sequences": args.num_return_sequences,
        "no_repeat_ngram_size": args.no_repeat_ngram_size,
    }
    if args.generation_config is None:
        return generation

    path = Path(args.generation_config).resolve()
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) if path.suffix in {".yaml", ".yml"} else json.load(handle)
    generation.update(loaded or {})
    return generation


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--mode", type=str, choices=["greedy", "rerank"], default="greedy")
    parser.add_argument("--split", type=str, default="eval")
    parser.add_argument("--audio-dir", type=str, default=None)
    parser.add_argument("--caption-csv", type=str, default=None)
    parser.add_argument("--output-csv", type=str, default=None)
    parser.add_argument("--details-jsonl", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--allow-partial-checkpoint", action="store_true")
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--env-file", type=str, default=str(REPO_ROOT / ".env"))
    parser.add_argument("--generation-config", type=str, default=None)
    parser.add_argument("--max-length", type=int, default=32)
    parser.add_argument("--min-length", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--num-return-sequences", type=int, default=64)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=3)
    parser.add_argument("--clap-model", type=str, default="laion/clap-htsat-unfused")
    parser.add_argument("--decoder-weight", type=float, default=0.3)
    parser.add_argument("--clap-weight", type=float, default=0.7)
    parser.add_argument("--top-audio-keep", type=int, default=32)
    parser.add_argument("--top-pairwise-keep", type=int, default=12)
    parser.add_argument("--openai-model", type=str, default="gpt-4.1-mini")
    parser.add_argument("--no-llm", action="store_true")
    args = parser.parse_args()

    load_dotenv(Path(args.env_file).resolve())
    config = load_config(Path(args.config).resolve())
    output_dir = Path(args.output_dir).resolve()
    output_csv = Path(args.output_csv).resolve() if args.output_csv else output_dir / f"{args.mode}_output.csv"
    details_jsonl = (
        Path(args.details_jsonl).resolve()
        if args.details_jsonl
        else output_dir / f"{args.mode}_details.jsonl"
    )

    audio_dir, caption_csv = resolve_dataset_paths(
        config=config,
        split=args.split,
        audio_dir=args.audio_dir,
        caption_csv=args.caption_csv,
    )
    print(f"[data] audio_dir={audio_dir}", flush=True)
    print(f"[data] caption_csv={caption_csv}", flush=True)

    device = choose_device(args.device)
    model = load_model(
        config=config,
        checkpoint_path=Path(args.checkpoint).resolve(),
        device=device,
        strict=not args.allow_partial_checkpoint,
    )
    dataset = build_dataset(audio_dir=audio_dir, caption_csv=caption_csv)
    dataloader = build_dataloader(
        config=config,
        dataset=dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    tokenizer = dataloader.collate_fn.tokenizer
    generation_config = load_generation_config(args)

    pipeline = None
    if args.mode == "rerank":
        clap_scorer = ClapSimilarityScorer(model_name=args.clap_model, device=device)
        pipeline = CaptionRerankingPipeline(
            model=model,
            tokenizer=tokenizer,
            clap_scorer=clap_scorer,
            device=device,
            generation_config=generation_config,
            decoder_weight=args.decoder_weight,
            clap_weight=args.clap_weight,
            top_audio_keep=args.top_audio_keep,
            top_pairwise_keep=args.top_pairwise_keep,
            openai_model=args.openai_model,
            use_llm=not args.no_llm,
            show_sampled_captions=True,
        )

    completed = load_completed(output_csv)
    processed_now = 0
    seen = 0
    with torch.no_grad():
        for batch_index, batch in enumerate(dataloader):
            pending_indices = [
                index
                for index, sample_id in enumerate(batch["sample_ids"])
                if sample_id not in completed
            ]
            if not pending_indices:
                continue
            pending_files = [batch["sample_ids"][index] for index in pending_indices]
            print(
                f"[infer] batch={batch_index + 1} files={', '.join(pending_files)}",
                flush=True,
            )

            if args.mode == "greedy":
                token_ids = model.generate(
                    waveforms=batch["waveforms"].to(device),
                    waveform_lengths=batch["waveform_lengths"].to(device),
                    sample_rates=batch["sample_rates"].to(device),
                    max_length=generation_config["max_length"],
                    min_length=generation_config.get("min_length", 0),
                    do_sample=False,
                    num_return_sequences=1,
                    no_repeat_ngram_size=generation_config.get("no_repeat_ngram_size", 0),
                )
                captions = tokenizer.batch_decode(
                    token_ids.detach().cpu(),
                    skip_special_tokens=True,
                )
                for index, caption in enumerate(captions):
                    sample_id = batch["sample_ids"][index]
                    if sample_id in completed:
                        continue
                    caption = normalize_caption(caption)
                    append_prediction(output_csv, sample_id, caption)
                    append_jsonl(
                        details_jsonl,
                        {
                            "audio_file": sample_id,
                            "mode": args.mode,
                            "caption_predicted": caption,
                        },
                    )
                    completed.add(sample_id)
                    processed_now += 1
                    seen += 1
                    print(f"[done] {sample_id}: {caption}", flush=True)
                    if args.max_items is not None and seen >= args.max_items:
                        return
            else:
                assert pipeline is not None
                for result in pipeline.process_batch(batch):
                    if result.sample_id in completed:
                        continue
                    append_prediction(output_csv, result.sample_id, result.final_caption)
                    record = CaptionRerankingPipeline.result_to_dict(result)
                    record["mode"] = args.mode
                    append_jsonl(details_jsonl, record)
                    completed.add(result.sample_id)
                    processed_now += 1
                    seen += 1
                    print(f"[done] {result.sample_id}: {result.final_caption}", flush=True)
                    if args.max_items is not None and seen >= args.max_items:
                        return

    print(
        f"[complete] wrote {processed_now} new rows to {output_csv}; "
        f"details={details_jsonl}",
        flush=True,
    )


if __name__ == "__main__":
    main()
