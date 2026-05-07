import argparse
import sys
import time
from pathlib import Path

import lightning as L
import yaml
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from aac_baseline.data.datamodule import ClothoDataModule
from aac_baseline.models.captioner import DCASE24BaselineCaptioner
from aac_baseline.training.callbacks import WandbValidationSamplesCallback
from aac_baseline.training.lightning_module import AudioCaptioningLightningModule


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def build_logger(config: dict):
    logging_cfg = config["logging"]
    save_dir = logging_cfg["save_dir"]

    if logging_cfg.get("use_wandb", True):
        return WandbLogger(
            project=logging_cfg["project"],
            entity=logging_cfg.get("entity"),
            name=logging_cfg["run_name"],
            save_dir=save_dir,
            log_model=False,
        )

    return TensorBoardLogger(save_dir=save_dir, name=logging_cfg["run_name"])


def log_run_configuration(logger, config: dict, config_path: Path) -> None:
    """
    Record the run configuration in the experiment logger.

    For W&B we do two things:
    - store the parsed config as structured run config for filtering/comparison
    - upload the exact YAML file so the original experiment definition is preserved
    """

    if logger is None:
        return

    logger.log_hyperparams(config)

    if not isinstance(logger, WandbLogger):
        return

    experiment = logger.experiment
    experiment.config.update(
        {
            **config,
            "config_path": str(config_path),
        },
        allow_val_change=True,
    )
    experiment.summary["config_path"] = str(config_path)

    if hasattr(experiment, "save"):
        experiment.save(str(config_path), policy="now")


def log_startup_step(start_time: float, message: str) -> None:
    elapsed = time.perf_counter() - start_time
    print(f"[startup +{elapsed:7.2f}s] {message}", flush=True)


def main() -> None:
    start_time = time.perf_counter()
    log_startup_step(start_time, "parsing command line arguments")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    log_startup_step(start_time, "loading YAML config")
    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    log_startup_step(start_time, f"loaded config from {config_path}")

    log_startup_step(start_time, "seeding libraries")
    L.seed_everything(config.get("seed", 42), workers=True)
    log_startup_step(start_time, "seed setup complete")

    log_startup_step(start_time, "building datamodule")
    datamodule = ClothoDataModule(config["data"])
    log_startup_step(start_time, "datamodule created")

    log_startup_step(start_time, "building captioner model")
    model = DCASE24BaselineCaptioner(config["model"])
    log_startup_step(start_time, "captioner model created")

    log_startup_step(start_time, "wrapping model in LightningModule")
    lightning_module = AudioCaptioningLightningModule(
        model=model,
        training_config=config["training"],
    )
    log_startup_step(start_time, "LightningModule created")

    log_startup_step(start_time, "initializing experiment logger")
    logger = build_logger(config)
    log_startup_step(start_time, f"logger ready: {type(logger).__name__}")

    log_startup_step(start_time, "logging run configuration")
    log_run_configuration(logger, config, config_path)
    log_startup_step(start_time, "run configuration logged")

    log_startup_step(start_time, "building callbacks")
    callbacks = [
        ModelCheckpoint(
            dirpath=Path(config["logging"]["save_dir"]) / "checkpoints",
            filename="aac-epoch{epoch:02d}-valloss{val_loss:.4f}",
            save_top_k=3,
            monitor="val_loss",
            mode="min",
        ),
        WandbValidationSamplesCallback(
            num_samples=config["training"].get("validation_preview_count", 4)
        ),
        LearningRateMonitor(logging_interval="step"),
    ]
    log_startup_step(start_time, "callbacks created")

    log_startup_step(start_time, "creating Lightning trainer")
    trainer = L.Trainer(
        accelerator=config["training"].get("accelerator", "auto"),
        devices=config["training"].get("devices", "auto"),
        strategy=config["training"].get("strategy", "auto"),
        precision=config["training"].get("precision", "16-mixed"),
        max_epochs=config["training"].get("max_epochs", 10),
        accumulate_grad_batches=config["training"].get("accumulate_grad_batches", 1),
        gradient_clip_val=config["training"].get("grad_clip_val", 1.0),
        log_every_n_steps=config["training"].get("log_every_n_steps", 10),
        val_check_interval=config["training"].get("val_check_interval", 1.0),
        limit_val_batches=config["training"].get("limit_val_batches", 1.0),
        logger=logger,
        callbacks=callbacks,
    )
    log_startup_step(start_time, "trainer created")

    log_startup_step(start_time, "starting trainer.fit()")
    trainer.fit(lightning_module, datamodule=datamodule)
    log_startup_step(start_time, "trainer.fit() returned")


if __name__ == "__main__":
    main()