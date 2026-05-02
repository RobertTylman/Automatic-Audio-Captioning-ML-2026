from pathlib import Path
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
        sample_rate: int,
        caption_mode: str = "random",
    ) -> None:
        self.audio_dir = Path(audio_dir)
        self.caption_csv = Path(caption_csv)
        self.sample_rate = sample_rate
        self.caption_mode = caption_mode

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

    def _load_audio(self, audio_path: str) -> torch.Tensor:
        waveform, sample_rate = torchaudio.load(audio_path)

        # The model expects mono audio. If a file is stereo, we average channels.
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        if sample_rate != self.sample_rate:
            waveform = torchaudio.functional.resample(
                waveform, orig_freq=sample_rate, new_freq=self.sample_rate
            )

        return waveform.squeeze(0)

    def __getitem__(self, index: int) -> Dict:
        example = self.examples[index]
        caption_text = self._select_caption(example)
        waveform = self._load_audio(example["audio_path"])

        return {
            "sample_id": example["file_name"],
            "audio_path": example["audio_path"],
            "waveform": waveform,
            "num_samples": int(waveform.numel()),
            "caption_text": caption_text,
            "all_captions": example["captions"],
            "sample_rate": self.sample_rate,
        }
