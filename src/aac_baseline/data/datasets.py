import json
from pathlib import Path
import time
from collections.abc import Sequence
from typing import Dict, List

import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset


def _clean_caption(value) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    return text


def _random_fallback_index(dataset_length: int, current_index: int) -> int:
    if dataset_length <= 1:
        return current_index
    next_index = torch.randint(dataset_length - 1, (1,)).item()
    if next_index >= current_index:
        next_index += 1
    return next_index


class ClothoCaptionDataset(Dataset):
    """
    This dataset deliberately returns a canonical representation of the sample:

    - waveform
    - sample rate
    - caption text
    - metadata

    Maybe the idea is that ee do not run encoder-specific preprocessing here. 
    Each encoder adapter should own its own preprocessing logic so the dataset
    stays reusable when we plug in new encoders.
    """

    def __init__(
        self,
        audio_dir: str,
        caption_csv: str,
        caption_mode: str = "random",
        stable_audio_augment: dict | None = None,
    ) -> None:
        self.audio_dir = Path(audio_dir)
        self.caption_csv = Path(caption_csv)
        self.caption_mode = caption_mode
        self._logged_example = False
        self.stable_audio_augment = stable_audio_augment or {}
        self.use_stable_audio_augment = bool(
            self.stable_audio_augment.get("enabled", False)
        )
        self.generated_audio_probability = float(
            self.stable_audio_augment.get("generated_audio_probability", 0.5)
        )
        self.generated_audio_by_file: dict[str, list[dict]] = {}
        self._load_error_count = 0

        dataframe = pd.read_csv(self.caption_csv)
        self.examples: List[Dict] = []

        for _, row in dataframe.iterrows():
            captions = [
                caption
                for idx in range(1, 6)
                if (caption := _clean_caption(row.get(f"caption_{idx}"))) is not None
            ]
            if not captions:
                continue
            audio_path = self.audio_dir / row["file_name"]
            if not audio_path.is_file():
                continue
            sample = {
                "file_name": row["file_name"],
                "audio_path": str(audio_path),
                "captions": captions,
            }

            if caption_mode == "all":
                for caption in captions:
                    self.examples.append({**sample, "caption_text": caption})
            else:
                self.examples.append(sample)

        if not self.examples:
            raise ValueError(
                f"No valid Clotho examples found for csv={self.caption_csv} audio_dir={self.audio_dir}"
            )

        if self.use_stable_audio_augment:
            self.generated_audio_by_file = self._load_generated_audio_index()

        print(
            f"[data] built ClothoCaptionDataset mode={self.caption_mode} "
            f"csv={self.caption_csv} examples={len(self.examples)}",
            flush=True,
        )
        if self.use_stable_audio_augment:
            total_generated = sum(
                len(paths) for paths in self.generated_audio_by_file.values()
            )
            print(
                "[data] stable audio augment enabled "
                f"files_with_generated_audio={len(self.generated_audio_by_file)} "
                f"generated_audio_files={total_generated} "
                f"probability={self.generated_audio_probability}",
                flush=True,
            )

    def __len__(self) -> int:
        return len(self.examples)

    def _load_generated_audio_index(self) -> dict[str, list[dict]]:
        root_value = self.stable_audio_augment.get("run_dir") or self.stable_audio_augment.get("root_dir")
        if not root_value:
            raise ValueError(
                "stable_audio_augment.enabled=true requires `run_dir` or `root_dir`."
            )

        root_dir = Path(root_value).expanduser()
        if not root_dir.exists():
            raise FileNotFoundError(f"Stable Audio augment directory does not exist: {root_dir}")

        allowed_splits = self.stable_audio_augment.get("splits")
        if allowed_splits is not None:
            allowed_splits = {str(split) for split in allowed_splits}

        known_files = {
            example["file_name"]
            for example in self.examples
        }
        generated_audio_by_file: dict[str, list[dict]] = {}
        manifest_paths = sorted(root_dir.glob("clips/*/*/manifest.json"))
        if not manifest_paths:
            manifest_paths = sorted(root_dir.glob("**/manifest.json"))

        for manifest_path in manifest_paths:
            try:
                with manifest_path.open("r", encoding="utf-8") as handle:
                    rows = json.load(handle)
            except (OSError, json.JSONDecodeError):
                continue

            if not isinstance(rows, list):
                continue

            for row in rows:
                if not isinstance(row, dict):
                    continue
                file_name = row.get("file_name")
                split = row.get("split")
                if file_name not in known_files:
                    continue
                if allowed_splits is not None and str(split) not in allowed_splits:
                    continue

                output_path = Path(str(row.get("output_path", ""))).expanduser()
                if not output_path.is_absolute():
                    output_path = (manifest_path.parent / output_path).resolve()
                if not output_path.is_file():
                    continue

                generated_audio_by_file.setdefault(file_name, []).append(
                    {
                        "audio_path": str(output_path),
                        "caption_index": row.get("caption_index"),
                        "variant_index": row.get("variant_index"),
                        "seed": row.get("seed"),
                    }
                )

        require_generated = bool(
            self.stable_audio_augment.get("require_generated_for_all", False)
        )
        if require_generated:
            missing_files = sorted(known_files.difference(generated_audio_by_file))
            if missing_files:
                raise ValueError(
                    "Missing Stable Audio augmentations for "
                    f"{len(missing_files)} files. First missing: {missing_files[:8]}"
                )

        return generated_audio_by_file

    def _select_caption(self, example: Dict) -> str:
        if self.caption_mode == "random":
            random_index = torch.randint(len(example["captions"]), (1,)).item()
            return example["captions"][random_index]

        if self.caption_mode == "first":
            return example["captions"][0]

        if self.caption_mode == "all":
            return example["caption_text"]

        raise ValueError(f"Unsupported caption_mode: {self.caption_mode}")

    def _load_audio(self, audio_path: str) -> tuple[torch.Tensor, int]:
        waveform, sample_rate = torchaudio.load(audio_path)

        # We collapse to mono here because all current encoder branches are
        # single-channel. Sample-rate conversion is intentionally NOT done here.
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        return waveform.squeeze(0), int(sample_rate)

    def _select_audio(self, example: Dict) -> tuple[str, str, dict | None]:
        generated_choices = self.generated_audio_by_file.get(example["file_name"], [])
        if not self.use_stable_audio_augment or not generated_choices:
            return example["audio_path"], "original", None

        use_generated = (
            torch.rand(1).item() < self.generated_audio_probability
        )
        if not use_generated:
            return example["audio_path"], "original", None

        choice_index = torch.randint(len(generated_choices), (1,)).item()
        generated = generated_choices[choice_index]
        return generated["audio_path"], "stable_audio_open", generated

    def __getitem__(self, index: int) -> Dict:
        last_error = None
        for _ in range(8):
            try:
                return self._getitem_impl(index)
            except Exception as exc:
                last_error = exc
                self._load_error_count += 1
                if self._load_error_count <= 5:
                    print(
                        f"[data] failed to load Clotho item index={index}: {exc}. "
                        "Trying another item.",
                        flush=True,
                    )
                index = _random_fallback_index(len(self), index)
        raise RuntimeError("Could not load a valid Clotho item after retries.") from last_error

    def _getitem_impl(self, index: int) -> Dict:
        example = self.examples[index]
        caption_text = self._select_caption(example)
        audio_path, audio_source, generated_metadata = self._select_audio(example)
        load_start = time.perf_counter()
        if not self._logged_example:
            print(
                f"[data] loading first audio example path={audio_path}",
                flush=True,
            )
        waveform, sample_rate = self._load_audio(audio_path)
        if not self._logged_example:
            elapsed = time.perf_counter() - load_start
            print(
                f"[data] first audio loaded samples={waveform.numel()} "
                f"sample_rate={sample_rate} source={audio_source} took={elapsed:.2f}s",
                flush=True,
            )
            self._logged_example = True

        return {
            "sample_id": example["file_name"],
            "audio_path": audio_path,
            "source_audio_path": example["audio_path"],
            "audio_source": audio_source,
            "generated_audio_metadata": generated_metadata,
            "waveform": waveform,
            "num_samples": int(waveform.numel()),
            "caption_text": caption_text,
            "all_captions": example["captions"],
            "sample_rate": sample_rate,
        }


class AudioCapsCaptionDataset(Dataset):
    """
    AudioCaps train/val/eval loader with the same output contract as Clotho.

    Expected layout:
    - root_dir/train.csv
    - root_dir/audio/train/<youtube_id>_<start_time>.wav
    """

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        caption_csv: str | None = None,
        audio_dir: str | None = None,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.split = split
        self.caption_csv = (
            Path(caption_csv)
            if caption_csv is not None
            else self.root_dir / f"{split}.csv"
        )
        self.audio_dir = (
            Path(audio_dir)
            if audio_dir is not None
            else self.root_dir / "audio" / split
        )
        self._logged_example = False

        dataframe = pd.read_csv(self.caption_csv)
        required_columns = {"youtube_id", "start_time", "caption"}
        missing_columns = required_columns.difference(dataframe.columns)
        if missing_columns:
            raise ValueError(
                f"{self.caption_csv} is missing required columns: {sorted(missing_columns)}"
            )

        self.examples: list[dict] = []
        for _, row in dataframe.iterrows():
            caption = _clean_caption(row.get("caption"))
            if caption is None:
                continue
            start_time = int(row["start_time"])
            youtube_id = str(row["youtube_id"])
            file_name = f"{youtube_id}_{start_time}.wav"
            audio_path = self.audio_dir / file_name
            if not audio_path.is_file():
                continue
            sample_id = str(row.get("audiocap_id", file_name))
            self.examples.append(
                {
                    "sample_id": sample_id,
                    "file_name": file_name,
                    "audio_path": str(audio_path),
                    "captions": [caption],
                    "caption_text": caption,
                }
            )

        if not self.examples:
            raise ValueError(
                f"No AudioCaps examples found for csv={self.caption_csv} audio_dir={self.audio_dir}"
            )

        print(
            f"[data] built AudioCapsCaptionDataset split={self.split} "
            f"csv={self.caption_csv} examples={len(self.examples)}",
            flush=True,
        )
        self._load_error_count = 0

    def __len__(self) -> int:
        return len(self.examples)

    def _load_audio(self, audio_path: str) -> tuple[torch.Tensor, int]:
        waveform, sample_rate = torchaudio.load(audio_path)
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        return waveform.squeeze(0), int(sample_rate)

    def __getitem__(self, index: int) -> Dict:
        last_error = None
        for _ in range(8):
            try:
                return self._getitem_impl(index)
            except Exception as exc:
                last_error = exc
                self._load_error_count += 1
                if self._load_error_count <= 5:
                    print(
                        f"[data] failed to load AudioCaps item index={index}: {exc}. "
                        "Trying another item.",
                        flush=True,
                    )
                index = _random_fallback_index(len(self), index)
        raise RuntimeError("Could not load a valid AudioCaps item after retries.") from last_error

    def _getitem_impl(self, index: int) -> Dict:
        example = self.examples[index]
        load_start = time.perf_counter()
        if not self._logged_example:
            print(
                f"[data] loading first AudioCaps example path={example['audio_path']}",
                flush=True,
            )
        waveform, sample_rate = self._load_audio(example["audio_path"])
        if not self._logged_example:
            elapsed = time.perf_counter() - load_start
            print(
                f"[data] first AudioCaps audio loaded samples={waveform.numel()} "
                f"sample_rate={sample_rate} took={elapsed:.2f}s",
                flush=True,
            )
            self._logged_example = True

        return {
            "sample_id": example["sample_id"],
            "audio_path": example["audio_path"],
            "source_audio_path": example["audio_path"],
            "audio_source": "audiocaps",
            "waveform": waveform,
            "num_samples": int(waveform.numel()),
            "caption_text": example["caption_text"],
            "all_captions": example["captions"],
            "sample_rate": sample_rate,
        }


class ClothoChatGptMixupDataset(Dataset):
    """
    On-the-fly Clotho audio mixup with ChatGPT mixup captions.

    The JSON format follows the DCASE 2023 BEATs-Conformer-BART release:
    each row contains two source audio files and one or more `chatgpt_mixups`.
    """

    def __init__(
        self,
        audio_dir: str,
        mixup_json: str,
        rejected_json: str | None = None,
        mixup_normalize: bool = False,
    ) -> None:
        self.audio_dir = Path(audio_dir)
        self.mixup_json = Path(mixup_json)
        self.rejected_json = Path(rejected_json) if rejected_json else None
        self.mixup_normalize = mixup_normalize
        self._logged_example = False
        self._load_error_count = 0
        self.examples = self._load_examples()

        if not self.examples:
            raise ValueError(f"No Clotho mixup examples found in {self.mixup_json}")

        print(
            f"[data] built ClothoChatGptMixupDataset json={self.mixup_json} "
            f"examples={len(self.examples)}",
            flush=True,
        )

    def __len__(self) -> int:
        return len(self.examples)

    def _load_examples(self) -> list[dict]:
        with self.mixup_json.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        rows = payload.get("dataset", payload)
        if not isinstance(rows, list):
            raise ValueError(f"{self.mixup_json} must contain a list or a dataset list.")

        rejected_pairs = set()
        if self.rejected_json is not None and self.rejected_json.is_file():
            with self.rejected_json.open("r", encoding="utf-8") as handle:
                rejected_rows = json.load(handle)
            for row in rejected_rows:
                if isinstance(row, dict) and "idx" in row:
                    rejected_pairs.add(tuple(row["idx"]))

        examples = []
        for row_index, row in enumerate(rows):
            if not isinstance(row, dict):
                continue
            audio_files = row.get("audio_files")
            selected_pair = row.get("selected_pair", [row_index, row_index])
            captions = []
            for caption_index, caption in enumerate(row.get("chatgpt_mixups", [])):
                if (row_index, caption_index) in rejected_pairs:
                    continue
                cleaned_caption = _clean_caption(caption)
                if cleaned_caption is not None:
                    captions.append(cleaned_caption)

            if (
                not isinstance(audio_files, Sequence)
                or len(audio_files) != 2
                or not captions
            ):
                continue

            audio_a = self.audio_dir / str(audio_files[0])
            audio_b = self.audio_dir / str(audio_files[1])
            if not audio_a.is_file() or not audio_b.is_file():
                continue

            examples.append(
                {
                    "sample_id": f"clotho_mixup_{selected_pair[0]}_{selected_pair[1]}",
                    "audio_paths": [str(audio_a), str(audio_b)],
                    "captions": captions,
                }
            )
        return examples

    def _load_audio(self, audio_path: str) -> tuple[torch.Tensor, int]:
        waveform, sample_rate = torchaudio.load(audio_path)
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        return waveform.squeeze(0), int(sample_rate)

    def _mix_audio(
        self,
        audio_a: torch.Tensor,
        audio_b: torch.Tensor,
    ) -> torch.Tensor:
        if self.mixup_normalize:
            audio_a = audio_a / audio_a.abs().max().clamp_min(1e-6)
            audio_b = audio_b / audio_b.abs().max().clamp_min(1e-6)

        energy_a = audio_a.pow(2).mean().clamp_min(1e-8)
        energy_b = audio_b.pow(2).mean().clamp_min(1e-8)
        mix_db_ratio = torch.empty(1).uniform_(-5.0, 5.0).item()
        mix_scale = torch.sqrt(energy_a / ((10 ** (mix_db_ratio / 10.0)) * energy_b))

        if audio_a.numel() >= audio_b.numel():
            mixed_audio = audio_a.clone()
            max_start = audio_a.numel() - audio_b.numel()
            start = torch.randint(max_start + 1, (1,)).item() if max_start > 0 else 0
            mixed_audio[start : start + audio_b.numel()] += mix_scale * audio_b
            return mixed_audio

        mixed_audio = audio_b.clone()
        max_start = audio_b.numel() - audio_a.numel()
        start = torch.randint(max_start + 1, (1,)).item() if max_start > 0 else 0
        mixed_audio[start : start + audio_a.numel()] += mix_scale * audio_a
        return mixed_audio

    def __getitem__(self, index: int) -> Dict:
        last_error = None
        for _ in range(8):
            try:
                return self._getitem_impl(index)
            except Exception as exc:
                last_error = exc
                self._load_error_count += 1
                if self._load_error_count <= 5:
                    print(
                        f"[data] failed to load Clotho mixup item index={index}: {exc}. "
                        "Trying another item.",
                        flush=True,
                    )
                index = _random_fallback_index(len(self), index)
        raise RuntimeError("Could not load a valid Clotho mixup item after retries.") from last_error

    def _getitem_impl(self, index: int) -> Dict:
        example = self.examples[index]
        caption_index = torch.randint(len(example["captions"]), (1,)).item()
        caption_text = example["captions"][caption_index]

        audio_a, sample_rate_a = self._load_audio(example["audio_paths"][0])
        audio_b, sample_rate_b = self._load_audio(example["audio_paths"][1])
        if sample_rate_a != sample_rate_b:
            audio_b = torchaudio.functional.resample(
                audio_b.unsqueeze(0),
                orig_freq=sample_rate_b,
                new_freq=sample_rate_a,
            ).squeeze(0)
        waveform = self._mix_audio(audio_a, audio_b)

        if not self._logged_example:
            print(
                f"[data] first Clotho mixup loaded paths={example['audio_paths']} "
                f"samples={waveform.numel()} sample_rate={sample_rate_a}",
                flush=True,
            )
            self._logged_example = True

        return {
            "sample_id": example["sample_id"],
            "audio_path": "|".join(example["audio_paths"]),
            "source_audio_path": "|".join(example["audio_paths"]),
            "audio_source": "clotho_chatgpt_mixup",
            "waveform": waveform,
            "num_samples": int(waveform.numel()),
            "caption_text": caption_text,
            "all_captions": example["captions"],
            "sample_rate": sample_rate_a,
        }


class WeightedDataset(Dataset):
    """
    Fixed-size weighted source sampler for multi-dataset training.
    """

    def __init__(
        self,
        datasets: dict[str, Dataset],
        weights: dict[str, float],
        epoch_size: int,
    ) -> None:
        if epoch_size <= 0:
            raise ValueError(f"epoch_size must be positive, got {epoch_size}")
        if not datasets:
            raise ValueError("WeightedDataset requires at least one dataset.")

        self.datasets = datasets
        self.source_names = list(datasets.keys())
        raw_weights = [
            float(weights.get(source_name, 1.0))
            for source_name in self.source_names
        ]
        if any(weight < 0 for weight in raw_weights) or sum(raw_weights) <= 0:
            raise ValueError(f"Invalid dataset weights: {weights}")

        self.weights = torch.tensor(raw_weights, dtype=torch.float)
        self.weights = self.weights / self.weights.sum()
        self.epoch_size = epoch_size

    def __len__(self) -> int:
        return self.epoch_size

    def __getitem__(self, index: int) -> Dict:
        source_index = torch.multinomial(self.weights, num_samples=1).item()
        source_name = self.source_names[source_index]
        dataset = self.datasets[source_name]
        item_index = torch.randint(len(dataset), (1,)).item()
        item = dataset[item_index]
        item["dataset_source"] = source_name
        return item