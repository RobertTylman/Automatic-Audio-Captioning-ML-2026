from pathlib import Path
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
    ) -> None:
        self.audio_dir = Path(audio_dir)
        self.caption_csv = Path(caption_csv)
        self.caption_mode = caption_mode
        self._logged_example = False

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

        print(
            f"[data] built ClothoCaptionDataset mode={self.caption_mode} "
            f"csv={self.caption_csv} examples={len(self.examples)}",
            flush=True,
        )

    def __len__(self) -> int:
        return len(self.examples)

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

    def __getitem__(self, index: int) -> Dict:
        example = self.examples[index]
        caption_text = self._select_caption(example)
        load_start = time.perf_counter()
        if not self._logged_example:
            print(
                f"[data] loading first audio example path={example['audio_path']}",
                flush=True,
            )
        waveform, sample_rate = self._load_audio(example["audio_path"])
        if not self._logged_example:
            elapsed = time.perf_counter() - load_start
            print(
                f"[data] first audio loaded samples={waveform.numel()} "
                f"sample_rate={sample_rate} took={elapsed:.2f}s",
                flush=True,
            )
            self._logged_example = True

        return {
            "sample_id": example["file_name"],
            "audio_path": example["audio_path"],
            "waveform": waveform,
            "num_samples": int(waveform.numel()),
            "caption_text": caption_text,
            "all_captions": example["captions"],
            "sample_rate": sample_rate,
        }