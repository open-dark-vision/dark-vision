import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
)
import os

from src.configs.experiments import kind_config_illumin as cfg  # noqa: I900
from src.datasets import LOLDecompositionDataModul  # noqa: I900
from src.models import LitKinD_illumina  # noqa: I900
from src.utils import divide_chunks, save_predictions  # noqa: I900


if __name__ == "__main__":

    # ILLUMINATION

    cfg = OmegaConf.structured(cfg)
    print(cfg.dataset.path_decom)

    if cfg.dataset.path_decom is not None:
        lol_dm_decom = LOLDecompositionDataModul(config=cfg.dataset)
    else:
        print("No data. Training cannot be performed.")

    model = LitKinD_illumina(config=cfg, save_images=3)

    callbacks = [
        RichProgressBar(),
        ModelCheckpoint(
            dirpath=os.path.dirname(cfg.checkpoint_path) if cfg.checkpoint_path is not None else None,
            monitor="val/loss_illumina",
            mode="min",
            save_top_k=2,
            save_last=True,
            auto_insert_metric_name=False,
            filename=cfg.name + "-" + cfg.model_name + "-{epoch:03d}-loss-{val/loss:.4f}",
        ),  
        LearningRateMonitor(logging_interval="step"),
    ]

    trainer = pl.Trainer(
        accelerator=cfg.device,
        devices=1,
        callbacks=callbacks,
        max_epochs=cfg.epochs,
    )


    if cfg.checkpoint_path is not None:
        print("Loading from checkpoint: ", cfg.checkpoint_path)

        checkpoint = torch.load(cfg.checkpoint_path)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()

    else:
        trainer.fit(model, lol_dm_decom)