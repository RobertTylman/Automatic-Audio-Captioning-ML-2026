import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from aac_baseline.data.collators import AudioCaptioningCollator
from aac_baseline.data.datasets import ClothoCaptionDataset
from aac_baseline.inference import CaptionRerankingPipeline
from aac_baseline.inference.reranking_pipeline import ClapSimilarityScorer
from aac_baseline.models.captioner import DCASE24BaselineCaptioner


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
            key = key.strip()
            value = value.strip().strip("\"'")
            os.environ.setdefault(key, value)

    if "OPENAI_API_KEY" not in os.environ and "OPENAI" in os.environ:
        os.environ["OPENAI_API_KEY"] = os.environ["OPENAI"]


def load_model(config: dict, checkpoint_path: Path, device: torch.device):
    model = DCASE24BaselineCaptioner(config["model"])
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    model_state = {}

    for key, value in state_dict.items():
        if key.startswith("model."):
            model_state[key.removeprefix("model.")] = value
        else:
            model_state[key] = value

    model.load_state_dict(model_state, strict=False)
    model.to(device)
    model.eval()
    return model


def build_dataloader(config: dict, split: str, batch_size: int) -> DataLoader:
    data_config = config["data"]
    if split == "train":
        audio_dir = data_config["train_audio_dir"]
        caption_csv = data_config["train_caption_csv"]
        caption_mode = data_config.get("train_caption_mode", "first")
    elif split == "val":
        audio_dir = data_config["val_audio_dir"]
        caption_csv = data_config["val_caption_csv"]
        caption_mode = data_config.get("val_caption_mode", "first")
    else:
        raise ValueError(f"Unsupported split: {split}")

    dataset = ClothoCaptionDataset(
        audio_dir=audio_dir,
        caption_csv=caption_csv,
        caption_mode=caption_mode,
    )
    collator = AudioCaptioningCollator(
        tokenizer_name=data_config["tokenizer_name"],
        max_caption_tokens=data_config["max_caption_tokens"],
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=data_config.get("num_workers", 0),
        collate_fn=collator,
        pin_memory=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--clap-model", type=str, default="laion/clap-htsat-unfused")
    parser.add_argument("--openai-model", type=str, default="gpt-4.1-mini")
    parser.add_argument("--top-audio-keep", type=int, default=32)
    parser.add_argument("--top-pairwise-keep", type=int, default=12)
    parser.add_argument("--env-file", type=str, default=str(REPO_ROOT / ".env"))
    args = parser.parse_args()

    load_dotenv(Path(args.env_file).resolve())
    config = load_config(Path(args.config).resolve())
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    model = load_model(
        config=config,
        checkpoint_path=Path(args.checkpoint).resolve(),
        device=device,
    )
    dataloader = build_dataloader(
        config=config,
        split=args.split,
        batch_size=args.batch_size,
    )
    clap_scorer = ClapSimilarityScorer(model_name=args.clap_model, device=device)
    pipeline = CaptionRerankingPipeline(
        model=model,
        tokenizer=dataloader.collate_fn.tokenizer,
        clap_scorer=clap_scorer,
        device=device,
        generation_config=config["training"].get("generation", {}),
        top_audio_keep=args.top_audio_keep,
        top_pairwise_keep=args.top_pairwise_keep,
        openai_model=args.openai_model,
    )

    details = []
    csv_rows = {"file_name": [], "caption_predicted": []}
    for batch_index, batch in enumerate(dataloader):
        if args.max_batches is not None and batch_index >= args.max_batches:
            break

        for result in pipeline.process_batch(batch):
            details.append(CaptionRerankingPipeline.result_to_dict(result))
            csv_rows["file_name"].append(result.sample_id)
            csv_rows["caption_predicted"].append(result.final_caption)

        print(f"[info] processed batch {batch_index + 1}")

    with (output_dir / "reranking_details.json").open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(details, indent=4))
        handle.write("\n")

    pd.DataFrame.from_dict(csv_rows).to_csv(
        output_dir / "llm_sap_summary_output.csv",
        index=False,
    )


if __name__ == "__main__":
    main()
