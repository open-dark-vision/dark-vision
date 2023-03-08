import pytorch_lightning as pl
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from src.configs.experiments import bridimo_config as cfg  # noqa: I900
from src.datasets import COCODataModule  # noqa: I900
from src.models import LitBriDiMo  # noqa: I900

if __name__ == "__main__":
    cfg = OmegaConf.structured(cfg)
    coco_dm = COCODataModule(config=cfg.dataset)

    model = LitBriDiMo(config=cfg)

    callbacks = [
        pl.callbacks.progress.TQDMProgressBar(),
        # RichProgressBar(),
        ModelCheckpoint(
            monitor="val/loss",
            mode="min",
            save_top_k=2,
            save_last=True,
            auto_insert_metric_name=False,
            filename=cfg.name + "-{epoch:03d}-loss-{val/loss:.4f}",
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
    trainer.fit(model, coco_dm)
