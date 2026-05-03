import argparse
import sys
from pathlib import Path

import lightning as L
import yaml
from lightning.pytorch.callbacks import ModelCheckpoint
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
            name=logging_cfg["run_name"],
            save_dir=save_dir,
            log_model=False,
        )

    return TensorBoardLogger(save_dir=save_dir, name=logging_cfg["run_name"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    config = load_config(config_path)

    L.seed_everything(config.get("seed", 42), workers=True)

    datamodule = ClothoDataModule(config["data"])
    model = DCASE24BaselineCaptioner(config["model"])
    lightning_module = AudioCaptioningLightningModule(
        model=model,
        training_config=config["training"],
    )

    logger = build_logger(config)
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
    ]

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

    trainer.fit(lightning_module, datamodule=datamodule)


if __name__ == "__main__":
    main()
