import time

from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities.rank_zero import rank_zero_only


class TrainingProgressCallback(Callback):
    def __init__(self, log_every_n_steps: int = 100) -> None:
        super().__init__()
        self.log_every_n_steps = max(1, int(log_every_n_steps))
        self.epoch_start_time = None

    @rank_zero_only
    def _print(self, message: str) -> None:
        print(message, flush=True)

    def on_train_start(self, trainer, pl_module) -> None:
        self._print(
            "[progress] train_start "
            f"epoch={trainer.current_epoch} global_step={trainer.global_step} "
            f"world_size={trainer.world_size} "
            f"num_devices={getattr(trainer, 'num_devices', 'unknown')} "
            f"strategy={type(trainer.strategy).__name__}"
        )
        logger = trainer.logger
        if trainer.is_global_zero and logger is not None and hasattr(logger, "experiment"):
            experiment = logger.experiment
            if hasattr(experiment, "log"):
                experiment.log(
                    {
                        "trainer/current_epoch": trainer.current_epoch,
                        "trainer/global_step": trainer.global_step,
                        "trainer/world_size": trainer.world_size,
                        "trainer/num_devices": getattr(trainer, "num_devices", 0),
                    },
                    step=trainer.global_step,
                )

    def on_train_epoch_start(self, trainer, pl_module) -> None:
        self.epoch_start_time = time.perf_counter()
        self._print(
            "[progress] epoch_start "
            f"epoch={trainer.current_epoch} global_step={trainer.global_step} "
            f"train_batches={trainer.num_training_batches} "
            f"val_batches={trainer.num_val_batches}"
        )

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx: int) -> None:
        if trainer.global_step == 0 or trainer.global_step % self.log_every_n_steps != 0:
            return
        self._print(
            "[progress] train_batch "
            f"epoch={trainer.current_epoch} batch_idx={batch_idx} "
            f"global_step={trainer.global_step}/{trainer.estimated_stepping_batches}"
        )

    def on_validation_epoch_start(self, trainer, pl_module) -> None:
        if trainer.sanity_checking:
            return
        self._print(
            "[progress] validation_start "
            f"epoch={trainer.current_epoch} global_step={trainer.global_step} "
            f"val_batches={trainer.num_val_batches}"
        )

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        elapsed = None
        if self.epoch_start_time is not None:
            elapsed = time.perf_counter() - self.epoch_start_time
        elapsed_text = "" if elapsed is None else f" elapsed_sec={elapsed:.1f}"
        self._print(
            "[progress] epoch_end "
            f"epoch={trainer.current_epoch} global_step={trainer.global_step}"
            f"{elapsed_text}"
        )

    def on_fit_end(self, trainer, pl_module) -> None:
        self._print(
            "[progress] fit_end "
            f"epoch={trainer.current_epoch} global_step={trainer.global_step}"
        )


class WandbValidationSamplesCallback(Callback):
    """
    Log a few validation examples to wandb after each validation epoch.

    """

    def __init__(self, num_samples: int = 4) -> None:
        super().__init__()
        self.num_samples = num_samples

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        if not trainer.is_global_zero:
            return

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
                    sample_rate=int(preview["sample_rates"][index].item()),
                ),
            )

        epoch_key = f"validation_samples_epoch_{trainer.current_epoch:03d}"
        experiment.log(
            {epoch_key: table},
            step=trainer.global_step,
            commit=False,
        )

        logged_epochs = experiment.summary.get("validation_sample_epochs", [])
        if trainer.current_epoch not in logged_epochs:
            logged_epochs = list(logged_epochs) + [trainer.current_epoch]

        experiment.summary["validation_sample_epochs"] = logged_epochs
        experiment.summary["latest_validation_samples_key"] = epoch_key