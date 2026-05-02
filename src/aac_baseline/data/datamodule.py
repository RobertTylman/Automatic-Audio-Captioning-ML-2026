import lightning as L
from torch.utils.data import DataLoader

from .collators import AudioCaptioningCollator
from .datasets import ClothoCaptionDataset


class ClothoDataModule(L.LightningDataModule):
    def __init__(self, data_config: dict) -> None:
        super().__init__()
        self.data_config = data_config
        self.collator = AudioCaptioningCollator(
            tokenizer_name=data_config["tokenizer_name"],
            max_caption_tokens=data_config["max_caption_tokens"],
        )

    def setup(self, stage: str | None = None) -> None:
        self.train_dataset = ClothoCaptionDataset(
            audio_dir=self.data_config["train_audio_dir"],
            caption_csv=self.data_config["train_caption_csv"],
            sample_rate=self.data_config["sample_rate"],
            caption_mode=self.data_config.get("train_caption_mode", "random"),
        )
        self.val_dataset = ClothoCaptionDataset(
            audio_dir=self.data_config["val_audio_dir"],
            caption_csv=self.data_config["val_caption_csv"],
            sample_rate=self.data_config["sample_rate"],
            caption_mode=self.data_config.get("val_caption_mode", "first"),
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.data_config["batch_size"],
            shuffle=True,
            num_workers=self.data_config["num_workers"],
            collate_fn=self.collator,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.data_config["batch_size"],
            shuffle=False,
            num_workers=self.data_config["num_workers"],
            collate_fn=self.collator,
            pin_memory=True,
        )
