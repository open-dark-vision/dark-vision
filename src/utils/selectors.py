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
from src.losses import SCILoss  # noqa: I900


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
            betas=config.betas,
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
    elif config.scheduler.name == Scheduler.KIND:
        scheduler1 = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[800, 1250, 1500], gamma=0.5)#torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1, total_iters=800)
        scheduler2 = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=0.1, total_iters=500)
        scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer=optimizer, schedulers=[scheduler1, scheduler2], milestones=[1500])
        
    else:
        raise NotImplementedError(
            f"Scheduler {config.scheduler.name} is not implemented."
        )
    return {
        "optimizer": optimizer,
        "lr_scheduler": {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": config.scheduler.frequency,
        },
    }


def get_loss(config: LossConfig, **kwargs):
    if config.name == Loss.L1:
        return nn.L1Loss(reduction=config.reduction)
    elif config.name == Loss.SCI:
        return SCILoss(finetune=kwargs["finetune"])
    else:
        raise NotImplementedError(f"Loss {config.name} is not implemented.")
