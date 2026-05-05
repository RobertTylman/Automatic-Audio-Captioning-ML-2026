from lightning.pytorch.callbacks import Callback


class WandbValidationSamplesCallback(Callback):
    """
    Log a few validation examples to wandb after each validation epoch.

    """
    def __init__(self, num_samples: int = 4, generation_config: dict | None = None) -> None:
        super().__init__()
        self.num_samples = num_samples
        self.generation_config = generation_config or {}

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        preview = pl_module.cached_validation_preview
        logger = trainer.logger

        if preview is None or logger is None or not hasattr(logger, "experiment"):
            return

        experiment = logger.experiment
        if not hasattr(experiment, "log"):
            return

        num_samples = min(self.num_samples, preview["waveforms"].size(0))
        predictions = pl_module.model.generate(
            waveforms=preview["waveforms"][:num_samples].to(pl_module.device),
            waveform_lengths=preview["waveform_lengths"][:num_samples].to(pl_module.device),
            sample_rates=preview["sample_rates"][:num_samples].to(pl_module.device),
            **self.generation_config,
        )
        decoded_predictions = preview["tokenizer"].batch_decode(
            predictions.detach().cpu(),
            skip_special_tokens=True,
        )

        try:
            import wandb
        except ImportError:
            return

        table = wandb.Table(columns=["reference", "prediction", "audio"])
        num_return_sequences = self.generation_config.get("num_return_sequences", 1)
        for index in range(num_samples):
            prediction_start = index * num_return_sequences
            prediction_end = prediction_start + num_return_sequences
            prediction = "\n".join(
                decoded_predictions[prediction_start:prediction_end]
            )
            table.add_data(
                preview["caption_texts"][index],
                prediction,
                wandb.Audio(
                    preview["waveforms"][index].detach().cpu().numpy(),
                    sample_rate=int(preview["sample_rates"][index].item()),
                ),
            )

        epoch_key = f"validation_samples_epoch_{trainer.current_epoch:03d}"
        experiment.log(
            {epoch_key: table},
            step=trainer.current_epoch,
            commit=False,
        )

        logged_epochs = experiment.summary.get("validation_sample_epochs", [])
        if trainer.current_epoch not in logged_epochs:
            logged_epochs = list(logged_epochs) + [trainer.current_epoch]

        experiment.summary["validation_sample_epochs"] = logged_epochs
        experiment.summary["latest_validation_samples_key"] = epoch_key
