import pytorch_lightning as pl
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from src.configs.experiments import bridimo_finetune_config as cfg  # noqa: I900
from src.datasets import LOLBriDiMoDataModule  # noqa: I900
from src.models import LitBriDiMo  # noqa: I900

if __name__ == "__main__":
    cfg = OmegaConf.structured(cfg)
    coco_dm = LOLBriDiMoDataModule(config=cfg.dataset)

    chk_path = (
        "./reproducibility/iogve6zm/checkpoints/bridimo-coco-013-loss-0.0020.ckpt"
    )
    model = LitBriDiMo.load_from_checkpoint(chk_path, config=cfg)

    # freeze weight of backbone except for the last two layers
    for name, param in model.backbone.named_parameters():
        if "outc" not in name and "up4" not in name:
            param.requires_grad = False

    callbacks = [
        pl.callbacks.progress.TQDMProgressBar(),
        ModelCheckpoint(
            monitor="val/psnr",
            mode="max",
            save_top_k=1,
            save_last=False,
            auto_insert_metric_name=False,
            filename=cfg.name + "-{epoch:03d}-psnr-{val/psnr:.4f}",
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
