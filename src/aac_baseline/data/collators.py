from typing import Dict, List

import torch
from transformers import AutoTokenizer


class AudioCaptioningCollator:
    """
    This collator keeps two jobs together:

    1. pad raw waveforms into a batch
    2. tokenize captions for the decoder

    The encoder adapters still receive raw waveforms, not precomputed features.
    """

    def __init__(self, tokenizer_name: str, max_caption_tokens: int) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
        self.max_caption_tokens = max_caption_tokens

    def __call__(self, batch: List[Dict]) -> Dict:
        max_audio_len = max(example["num_samples"] for example in batch)
        padded_waveforms = []
        waveform_lengths = []
        sample_rates = []

        for example in batch:
            waveform = example["waveform"]
            pad_amount = max_audio_len - waveform.numel()
            padded_waveforms.append(torch.nn.functional.pad(waveform, (0, pad_amount)))
            waveform_lengths.append(example["num_samples"])
            sample_rates.append(example["sample_rate"])

        tokenized = self.tokenizer(
            [example["caption_text"] for example in batch],
            padding=True,
            truncation=True,
            max_length=self.max_caption_tokens,
            return_tensors="pt",
        )

        labels = tokenized["input_ids"].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "waveforms": torch.stack(padded_waveforms, dim=0),
            "waveform_lengths": torch.tensor(waveform_lengths, dtype=torch.long),
            "sample_rates": torch.tensor(sample_rates, dtype=torch.long),
            "labels": labels,
            "decoder_attention_mask": tokenized["attention_mask"],
            "sample_ids": [example["sample_id"] for example in batch],
            "audio_paths": [example["audio_path"] for example in batch],
            "caption_texts": [example["caption_text"] for example in batch],
            "all_captions": [example["all_captions"] for example in batch],
            "tokenizer": self.tokenizer,
        }