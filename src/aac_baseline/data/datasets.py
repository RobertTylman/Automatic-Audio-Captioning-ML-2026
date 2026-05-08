from pathlib import Path
import json
import time
from typing import Dict, List

import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset


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

        dataframe = pd.read_csv(self.caption_csv)
        self.examples: List[Dict] = []

        for _, row in dataframe.iterrows():
            captions = [row[f"caption_{idx}"] for idx in range(1, 6)]
            sample = {
                "file_name": row["file_name"],
                "audio_path": str(self.audio_dir / row["file_name"]),
                "captions": captions,
            }

            if caption_mode == "all":
                for caption in captions:
                    self.examples.append({**sample, "caption_text": caption})
            else:
                self.examples.append(sample)

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