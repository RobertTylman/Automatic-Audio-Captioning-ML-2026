from lightning.pytorch.callbacks import Callback


class WandbValidationSamplesCallback(Callback):
    """
    Log a few validation examples to wandb after each validation epoch.

    """

    def __init__(self, num_samples: int = 4) -> None:
        super().__init__()
        self.num_samples = num_samples

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
        for index in range(num_samples):
            table.add_data(
                preview["caption_texts"][index],
                decoded_predictions[index],
                wandb.Audio(
                    preview["waveforms"][index].detach().cpu().numpy(),
                    sample_rate=preview["sample_rate"],
                ),
            )

        experiment.log({"validation_samples": table}, commit=False)
