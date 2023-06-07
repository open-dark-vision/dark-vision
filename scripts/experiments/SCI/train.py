import pytorch_lightning as pl
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
)
from pytorch_lightning.loggers import WandbLogger

from src.configs.experiments import sci_config as cfg  # noqa: I900
from src.datasets import SICEDataModule  # noqa: I900
from src.models import LitSCI  # noqa: I900

cfg = OmegaConf.structured(cfg)

model = LitSCI(config=cfg)
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
    gradient_clip_val=5.0,
)
trainer.fit(model, sice_dm)
