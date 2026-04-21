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
            labels=batch["labels"],
        )

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        output = self(batch)
        self.log("train_loss", output.loss, prog_bar=True, on_step=True, on_epoch=True)
        return output.loss

    def validation_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        output = self(batch)
        self.log("val_loss", output.loss, prog_bar=True, on_step=False, on_epoch=True)

        if batch_idx == 0:
            self.cached_validation_preview = {
                "waveforms": batch["waveforms"][:],
                "waveform_lengths": batch["waveform_lengths"][:],
                "caption_texts": batch["caption_texts"],
                "sample_rate": batch["sample_rate"],
                "tokenizer": batch["tokenizer"],
            }

        return output.loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            params=[parameter for parameter in self.parameters() if parameter.requires_grad],
            lr=self.training_config["learning_rate"],
            weight_decay=self.training_config["weight_decay"],
        )

        warmup_steps = self.training_config.get("warmup_steps", 0)
        max_steps = self.training_config.get("max_steps", 10000)

        def lr_lambda(current_step: int) -> float:
            if warmup_steps > 0 and current_step < warmup_steps:
                return float(current_step + 1) / float(warmup_steps)

            remaining_steps = max(max_steps - warmup_steps, 1)
            progress = float(current_step - warmup_steps) / float(remaining_steps)
            return max(0.1, 1.0 - progress)

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }
