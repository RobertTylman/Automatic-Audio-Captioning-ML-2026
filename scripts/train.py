import argparse
import os
import sys
import time
from pathlib import Path

import lightning as L
import torch
import yaml
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from aac_baseline.data.datamodule import ClothoDataModule
from aac_baseline.models.captioner import DCASE24BaselineCaptioner
from aac_baseline.training.callbacks import TrainingProgressCallback, WandbValidationSamplesCallback
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

    if inferred_global_rank() != 0 or logger is None:
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


def inferred_global_rank() -> int:
    for variable_name in ("RANK", "SLURM_PROCID", "LOCAL_RANK"):
        value = os.environ.get(variable_name)
        if value is not None:
            return int(value)
    return 0


def parse_devices_override(devices: str):
    return int(devices) if devices.isdigit() else devices


def load_model_weights_from_checkpoint(
    lightning_module: AudioCaptioningLightningModule,
    checkpoint_path: str,
    *,
    strict: bool = True,
) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    load_result = lightning_module.load_state_dict(state_dict, strict=strict)
    if load_result is not None:
        print(f"[checkpoint] load result: {load_result}", flush=True)


def main() -> None:
    start_time = time.perf_counter()
    log_startup_step(start_time, "parsing command line arguments")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument(
        "--ckpt-path",
        type=str,
        default=None,
        help="Optional Lightning checkpoint path to resume training from.",
    )
    parser.add_argument(
        "--init-from-checkpoint",
        type=str,
        default=None,
        help=(
            "Optional checkpoint path used only to initialize model weights. "
            "Unlike --ckpt-path, optimizer state and epoch counters are not restored."
        ),
    )
    parser.add_argument(
        "--devices",
        type=str,
        default=None,
        help="Optional override for training.devices, for example 2 or auto.",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default=None,
        help="Optional override for training.strategy, for example ddp or auto.",
    )
    args = parser.parse_args()

    log_startup_step(start_time, "loading YAML config")
    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    if args.devices is not None:
        config["training"]["devices"] = parse_devices_override(args.devices)
    if args.strategy is not None:
        config["training"]["strategy"] = args.strategy
    log_startup_step(start_time, f"loaded config from {config_path}")
    ckpt_path = args.ckpt_path or config.get("training", {}).get("resume_from_checkpoint")
    init_checkpoint_path = args.init_from_checkpoint or config.get("training", {}).get(
        "init_from_checkpoint"
    )
    if ckpt_path is not None and init_checkpoint_path is not None:
        raise ValueError("Use either --ckpt-path or --init-from-checkpoint, not both.")
    if ckpt_path is not None:
        ckpt_path = str(Path(ckpt_path).expanduser().resolve())
        log_startup_step(start_time, f"will resume training from checkpoint {ckpt_path}")
    if init_checkpoint_path is not None:
        init_checkpoint_path = str(Path(init_checkpoint_path).expanduser().resolve())
        log_startup_step(
            start_time,
            f"will initialize model weights from checkpoint {init_checkpoint_path}",
        )

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
    if init_checkpoint_path is not None:
        log_startup_step(start_time, "loading model weights from checkpoint")
        load_model_weights_from_checkpoint(
            lightning_module,
            init_checkpoint_path,
            strict=config["training"].get("strict_init_checkpoint_load", True),
        )
        log_startup_step(start_time, "model weights loaded from checkpoint")

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
            save_last=True,
            monitor="val_loss",
            mode="min",
        ),
        WandbValidationSamplesCallback(
            num_samples=config["training"].get("validation_preview_count", 4)
        ),
        TrainingProgressCallback(
            log_every_n_steps=config["training"].get("progress_log_every_n_steps", 100)
        ),
        LearningRateMonitor(logging_interval="step"),
    ]
    log_startup_step(start_time, "callbacks created")

    log_startup_step(start_time, "creating Lightning trainer")
    log_startup_step(
        start_time,
        "trainer requested "
        f"accelerator={config['training'].get('accelerator', 'auto')} "
        f"devices={config['training'].get('devices', 'auto')} "
        f"strategy={config['training'].get('strategy', 'auto')}",
    )
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
        num_sanity_val_steps=config["training"].get("num_sanity_val_steps", 0),
        enable_progress_bar=config["training"].get("enable_progress_bar", False),
        logger=logger,
        callbacks=callbacks,
    )
    log_startup_step(start_time, "trainer created")

    log_startup_step(start_time, "starting trainer.fit()")
    trainer.fit(lightning_module, datamodule=datamodule, ckpt_path=ckpt_path)
    log_startup_step(start_time, "trainer.fit() returned")


if __name__ == "__main__":
    main()