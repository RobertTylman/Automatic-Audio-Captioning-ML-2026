import lightning as L
import torch

from .freezing import apply_freeze_policy


class AudioCaptioningLightningModule(L.LightningModule):
    def __init__(self, model, training_config: dict) -> None:
        super().__init__()
        self.model = model
        self.training_config = training_config
        self.cached_validation_preview = None

        apply_freeze_policy(
            model=self.model,
            freeze_modules=training_config.get("freeze_modules", []),
        )

    def forward(self, batch: dict):
        return self.model(
            waveforms=batch["waveforms"],
            waveform_lengths=batch["waveform_lengths"],
            sample_rates=batch["sample_rates"],
            labels=batch["labels"],
        )

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        output = self(batch)
        self.log(
            "train_loss",
            output.loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            batch_size=batch["waveforms"].size(0),
        )
        return output.loss

    def validation_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        output = self(batch)
        self.log(
            "val_loss",
            output.loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=batch["waveforms"].size(0),
        )

        if batch_idx == 0:
            self.cached_validation_preview = {
                "waveforms": batch["waveforms"][:],
                "waveform_lengths": batch["waveform_lengths"][:],
                "sample_rates": batch["sample_rates"][:],
                "caption_texts": batch["caption_texts"],
                "tokenizer": batch["tokenizer"],
            }

        return output.loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            params=[parameter for parameter in self.parameters() if parameter.requires_grad],
            lr=self.training_config["learning_rate"],
            weight_decay=self.training_config["weight_decay"],
        )
        return optimizer