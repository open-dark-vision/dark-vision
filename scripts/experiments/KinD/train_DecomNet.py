import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
)
import os

from src.configs.experiments import kind_config_decom as cfg  # noqa: I900
from src.datasets import LOLDataModule # noqa: I900
from src.models import LitKinD_decom  # noqa: I900
from src.utils import divide_chunks, save_predictions_decom  # noqa: I900


if __name__ == "__main__":

    # DECOMPOSITION

    cfg = OmegaConf.structured(cfg)
    lol_dm = LOLDataModule(config=cfg.dataset)

    model = LitKinD_decom(config=cfg, save_images=3)

    callbacks = [
        RichProgressBar(),
        ModelCheckpoint(
            dirpath=os.path.dirname(cfg.checkpoint_path) if cfg.checkpoint_path is not None else None,
            monitor="val/loss_decom",
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
        print("Training...")
        trainer.fit(model, lol_dm)


    if cfg.save_predictions is not None:
        print("Saving predictions...")
        if cfg.save_predictions == "training":
            cfg.dataset.predict_on_train = True
            predictions_train = trainer.predict(model, datamodule=lol_dm)
            cfg.dataset.predict_on_train = False
            cfg.dataset.predict_on_val = True
            predictions_val = trainer.predict(model, datamodule=lol_dm)
            predictions = predictions_train + predictions_val
            paths_train = list(divide_chunks(lol_dm.images_names_train, cfg.dataset.batch_size))
            paths_val = list(divide_chunks(lol_dm.images_names_val, cfg.dataset.batch_size))
            paths = paths_train + paths_val
            save_predictions_decom (predictions, paths)
        else:
            print("Saving test predictions")
            cfg.dataset.predict_on_val = False
            cfg.dataset.predict_on_train = False
            predictions_test = trainer.predict(model, datamodule=lol_dm)
            paths = list(divide_chunks(lol_dm.images_names_test, cfg.dataset.batch_size))
            save_predictions_decom (predictions_test, paths, mode="test")
