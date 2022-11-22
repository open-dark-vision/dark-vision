from pathlib import Path

import pytorch_lightning as pl
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
)
from pytorch_lightning.loggers import WandbLogger

from src.configs.experiments import sci_finetune_config as cfg  # noqa: I900
from src.datasets import LOLDataModule  # noqa: I900
from src.models import LitSCI  # noqa: I900

weights = Path("reproducibility/2w7xfjbz/checkpoints/sci-lol-029-loss-1.9908.ckpt")

cfg = OmegaConf.structured(cfg)
lol_dm = LOLDataModule(config=cfg.dataset)

model = LitSCI(config=cfg, weights=weights)
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
