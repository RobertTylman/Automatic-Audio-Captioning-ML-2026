import lightning as L
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from .collators import AudioCaptioningCollator
from .datasets import (
    AudioCapsCaptionDataset,
    ClothoCaptionDataset,
    ClothoChatGptMixupDataset,
    WeightedDataset,
)


class ClothoDataModule(L.LightningDataModule):
    def __init__(self, data_config: dict) -> None:
        super().__init__()
        self.data_config = data_config
        self.train_collator = AudioCaptioningCollator(
            tokenizer_name=data_config["tokenizer_name"],
            max_caption_tokens=data_config["max_caption_tokens"],
            include_all_caption_labels=False,
        )
        self.val_collator = AudioCaptioningCollator(
            tokenizer_name=data_config["tokenizer_name"],
            max_caption_tokens=data_config["max_caption_tokens"],
            include_all_caption_labels=True,
        )
        self.collator = self.train_collator

    def setup(self, stage: str | None = None) -> None:
        print(f"[data] datamodule.setup(stage={stage})", flush=True)
        self.train_dataset = self._build_train_dataset()
        self.val_dataset = self._build_validation_dataset()
        print(
            f"[data] setup complete train={len(self.train_dataset)} val={len(self.val_dataset)}",
            flush=True,
        )

    def _build_legacy_train_dataset(self) -> Dataset:
        return ClothoCaptionDataset(
            audio_dir=self.data_config["train_audio_dir"],
            caption_csv=self.data_config["train_caption_csv"],
            caption_mode=self.data_config.get("train_caption_mode", "random"),
            stable_audio_augment=self.data_config.get("stable_audio_augment"),
        )

    def _build_legacy_validation_dataset(self) -> Dataset:
        return ClothoCaptionDataset(
            audio_dir=self.data_config["val_audio_dir"],
            caption_csv=self.data_config["val_caption_csv"],
            caption_mode=self.data_config.get("val_caption_mode", "first"),
        )

    def _build_clotho_train_source(self, source_config: dict) -> Dataset:
        return ClothoCaptionDataset(
            audio_dir=source_config.get("audio_dir", self.data_config["train_audio_dir"]),
            caption_csv=source_config.get("caption_csv", self.data_config["train_caption_csv"]),
            caption_mode=source_config.get(
                "caption_mode",
                self.data_config.get("train_caption_mode", "random"),
            ),
            stable_audio_augment=source_config.get(
                "stable_audio_augment",
                self.data_config.get("stable_audio_augment"),
            ),
        )

    def _build_audiocaps_source(self, source_config: dict) -> Dataset:
        return AudioCapsCaptionDataset(
            root_dir=source_config["root_dir"],
            split=source_config.get("split", "train"),
            caption_csv=source_config.get("caption_csv"),
            audio_dir=source_config.get("audio_dir"),
        )

    def _build_clotho_mixup_source(self, source_config: dict) -> Dataset:
        return ClothoChatGptMixupDataset(
            audio_dir=source_config.get("audio_dir", self.data_config["train_audio_dir"]),
            mixup_json=source_config["mixup_json"],
            rejected_json=source_config.get("rejected_json"),
            mixup_normalize=source_config.get("mixup_normalize", False),
        )

    def _build_train_sources(self) -> dict[str, Dataset]:
        train_sources = self.data_config.get("train_sources")
        if not train_sources:
            return {"clotho": self._build_legacy_train_dataset()}

        datasets = {}
        for source_name, source_config in train_sources.items():
            source_config = source_config or {}
            if not source_config.get("enabled", False):
                continue

            if source_name == "clotho":
                datasets[source_name] = self._build_clotho_train_source(source_config)
            elif source_name == "audiocaps":
                datasets[source_name] = self._build_audiocaps_source(source_config)
            elif source_name in {"clotho_mixup", "chatgpt_mixup"}:
                datasets[source_name] = self._build_clotho_mixup_source(source_config)
            else:
                raise ValueError(f"Unsupported train source: {source_name}")

        if not datasets:
            raise ValueError("No enabled training sources were configured.")

        return datasets

    def _build_train_dataset(self) -> Dataset:
        datasets = self._build_train_sources()
        if len(datasets) == 1:
            return next(iter(datasets.values()))

        train_mix = self.data_config.get("train_mix", {})
        mode = train_mix.get("mode", "concat")
        if mode == "concat":
            return ConcatDataset(list(datasets.values()))

        if mode == "weighted":
            return WeightedDataset(
                datasets=datasets,
                weights=train_mix.get("weights", {}),
                epoch_size=int(train_mix["epoch_size"]),
            )

        raise ValueError(f"Unsupported train_mix mode: {mode}")

    def _build_validation_dataset(self) -> Dataset:
        validation_source = self.data_config.get("validation_source", "clotho")
        validation_sources = self.data_config.get("validation_sources", {})
        train_sources = self.data_config.get("train_sources", {})

        if validation_source == "clotho":
            return self._build_legacy_validation_dataset()

        if validation_source == "audiocaps":
            source_config = (
                validation_sources.get("audiocaps")
                or train_sources.get("audiocaps")
                or {}
            )
            if not source_config:
                raise ValueError(
                    "validation_source='audiocaps' requires "
                    "data.validation_sources.audiocaps or data.train_sources.audiocaps."
                )
            return self._build_audiocaps_source(source_config)

        raise ValueError(f"Unsupported validation_source: {validation_source}")

    def train_dataloader(self) -> DataLoader:
        print(
            f"[data] building train dataloader batch_size={self.data_config['batch_size']} "
            f"num_workers={self.data_config['num_workers']} pin_memory=True",
            flush=True,
        )
        return DataLoader(
            self.train_dataset,
            batch_size=self.data_config["batch_size"],
            shuffle=True,
            num_workers=self.data_config["num_workers"],
            collate_fn=self.train_collator,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        print(
            f"[data] building val dataloader batch_size={self.data_config['batch_size']} "
            f"num_workers={self.data_config['num_workers']} pin_memory=True",
            flush=True,
        )
        return DataLoader(
            self.val_dataset,
            batch_size=self.data_config["batch_size"],
            shuffle=False,
            num_workers=self.data_config["num_workers"],
            collate_fn=self.val_collator,
            pin_memory=True,
        )