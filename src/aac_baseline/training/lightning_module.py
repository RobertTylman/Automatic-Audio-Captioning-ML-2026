import lightning as L
import torch

from .freezing import apply_freeze_policy
from .metrics import compute_caption_metrics


class AudioCaptioningLightningModule(L.LightningModule):
    def __init__(self, model, training_config: dict) -> None:
        super().__init__()
        self.model = model
        self.training_config = training_config
        self.cached_validation_preview = None
        self.compute_validation_caption_metrics = training_config.get(
            "compute_caption_metrics",
            False,
        )
        self.caption_metric_batches = training_config.get("caption_metric_batches", 0)
        self.generation_max_length = training_config.get("generation_max_length", 32)

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

        if (
            self.compute_validation_caption_metrics
            and batch_idx < self.caption_metric_batches
        ):
            predictions = self.model.generate(
                waveforms=batch["waveforms"],
                waveform_lengths=batch["waveform_lengths"],
                sample_rates=batch["sample_rates"],
                max_length=self.generation_max_length,
            )
            decoded_predictions = batch["tokenizer"].batch_decode(
                predictions.detach().cpu(),
                skip_special_tokens=True,
            )
            metrics = compute_caption_metrics(
                predictions=decoded_predictions,
                references=batch["all_captions"],
            )
            for metric_name, metric_value in metrics.items():
                self.log(
                    f"val_{metric_name}",
                    metric_value,
                    prog_bar=metric_name in {"bleu4", "rouge_l"},
                    on_step=False,
                    on_epoch=True,
                    batch_size=len(decoded_predictions),
                )

        return output.loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            params=[parameter for parameter in self.parameters() if parameter.requires_grad],
            lr=self.training_config["learning_rate"],
            weight_decay=self.training_config["weight_decay"],
        )
        return optimizer