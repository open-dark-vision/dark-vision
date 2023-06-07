import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
)
import os

from src.configs.experiments import kind_config_finale as cfg  # noqa: I900
from src.datasets import LOLDataModule # noqa: I900
from src.models import  LitKinD  # noqa: I900
from src.utils import divide_chunks, save_predictions  # noqa: I900


if __name__ == "__main__":

    # FINALE CONNECTION

    cfg = OmegaConf.structured(cfg)
    lol_dm = LOLDataModule(config=cfg.dataset)
    model = LitKinD(config=cfg)

    if cfg.checkpoint_path is not None:
        print("Loading from checkpoint: ", cfg.checkpoint_path)
        checkpoint_decom = torch.load(cfg.checkpoint_path + "decom.ckpt", map_location=torch.device(cfg.device))
        model.decom_net.load_state_dict(checkpoint_decom["state_dict"])
        model.decom_net.eval()
        checkpoint_illumina = torch.load(cfg.checkpoint_path + "illumina.ckpt", map_location=torch.device(cfg.device))
        model.illum_net.load_state_dict(checkpoint_illumina["state_dict"])
        model.illum_net.eval()
        checkpoint_restore = torch.load(cfg.checkpoint_path + "restore.ckpt", map_location=torch.device(cfg.device))
        model.restore_net.load_state_dict(checkpoint_restore["state_dict"])
        model.restore_net.eval()
    else:
        print("There is no checkpoint")

    callbacks = [
        RichProgressBar(),
        LearningRateMonitor(logging_interval="step"),
    ]

    trainer = pl.Trainer(
        accelerator=cfg.device,
        devices=1,
        callbacks=callbacks,
        max_epochs=cfg.epochs,
    )


    predictions_test = trainer.predict(model, datamodule=lol_dm)
    paths = list(divide_chunks(lol_dm.images_names_test, cfg.dataset.batch_size))

    save_predictions(predictions_test, paths, mode="test")
