from pathlib import Path

import pytorch_lightning as pl
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
)
from pytorch_lightning.loggers import WandbLogger

from src.configs.experiments import iat_finetune_config as cfg  # noqa: I900
from src.datasets import LOLDataModule  # noqa: I900
from src.models import LitIAT  # noqa: I900

run_path = Path(
    "reproducibility/1degqnca/checkpoints/iat-lol-patches-330-loss-0.0754.ckpt"
)

cfg = OmegaConf.structured(cfg)
lol_dm = LOLDataModule(config=cfg.dataset)

model = LitIAT.load_from_checkpoint(run_path, config=cfg)
callbacks = [
    RichProgressBar(),
    ModelCheckpoint(
        monitor="val/loss",
        mode="min",
        save_top_k=2,
        save_last=True,
        auto_insert_metric_name=False,
        filename=cfg.name + "-{epoch:03d}-loss-{val/loss:.4f}",
        save_weights_only=True,
    ),
    LearningRateMonitor(logging_interval="step"),
]
logger = WandbLogger(entity="dark-vision", project="reproducibility", name=cfg.name)

trainer = pl.Trainer(
    accelerator=cfg.device,
    devices=1,
    callbacks=callbacks,
    logger=logger,
    max_epochs=cfg.epochs,
)
trainer.fit(model, lol_dm)
