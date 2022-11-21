import pytorch_lightning as pl
import torch
import torch.nn as nn

from src.configs.base import (  # noqa: I900
    Loss,
    LossConfig,
    Optimizer,
    OptimizerConfig,
    Scheduler,
)


def get_optimizers(model: pl.LightningModule, config: OptimizerConfig):
    if config.name == Optimizer.ADAMW:
        optimizer = torch.optim.AdamW(
            params=model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )
    elif config.name == Optimizer.ADAM:
        optimizer = torch.optim.Adam(
            params=model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )
    elif config.name == Optimizer.SGD:
        optimizer = torch.optim.SGD(
            params=model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )
    else:
        raise NotImplementedError(f"Optimizer {config.name} is not implemented.")

    if config.scheduler is None or config.scheduler.name == Scheduler.CONSTANT:
        return optimizer

    if config.scheduler.name == Scheduler.ONE_CYCLE:
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=optimizer,
            max_lr=config.lr,
            total_steps=model.trainer.estimated_stepping_batches,
        )
    elif config.scheduler.name == Scheduler.COSINE:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer,
            T_max=model.trainer.estimated_stepping_batches,
        )
    else:
        raise NotImplementedError(
            f"Scheduler {config.scheduler.name} is not implemented."
        )
    print(scheduler, "HEREHRE")
    return {
        "optimizer": optimizer,
        "lr_scheduler": {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": config.scheduler.frequency,
            "name": "LR Scheduler",
        },
    }


def get_loss(config: LossConfig):
    if config.name == Loss.L1:
        return nn.L1Loss(reduction=config.reduction)
    else:
        raise NotImplementedError(f"Loss {config.name} is not implemented.")
