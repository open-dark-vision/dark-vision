from pathlib import Path

import pytorch_lightning as pl
import torch
import torchmetrics
from einops import einsum, rearrange
from omegaconf import OmegaConf
from torch import nn

from src.models.IAT.global_net import GlobalNet  # noqa: I900
from src.models.IAT.local_net import LocalNet  # noqa: I900
from src.utils import get_loss, get_optimizers  # noqa: I900

# based on https://github.com/cuiziteng/Illumination-Adaptive-Transformer


class IAT(nn.Module):
    def __init__(self, in_dim=3, task_type="lol", layers_type="ccc"):
        super().__init__()

        self.local_net = LocalNet(in_dim=in_dim, layers_type=layers_type)
        self.global_net = GlobalNet(in_channels=in_dim, task_type=task_type)

    def apply_color(self, image, ccm):
        image = einsum(image, ccm, "h w c, k c -> h w k")
        return torch.clamp(image, 1e-8, 1.0)

    def forward(self, img_low):
        mul, add = self.local_net(img_low)
        img_high = img_low * mul + add

        gamma, color = self.global_net(img_low)
        b = img_high.shape[0]
        img_high = rearrange(img_high, "b c h w -> b h w c")
        img_high = torch.stack(
            [
                self.apply_color(img_high[i, ...], color[i, ...]) ** gamma[i, :]
                for i in range(b)
            ],
            dim=0,
        )
        img_high = rearrange(img_high, "b h w c -> b c h w")

        return mul, add, img_high


class LitIAT(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.model = IAT(
            in_dim=config.model.in_dim,
            task_type=config.model.task_type,
            layers_type=config.model.layers_type,
        )
        self.loss_fn = get_loss(config.loss)

        self.train_psnr = torchmetrics.PeakSignalNoiseRatio()
        self.train_ssim = torchmetrics.StructuralSimilarityIndexMeasure()

        self.val_psnr = torchmetrics.PeakSignalNoiseRatio()
        self.val_ssim = torchmetrics.StructuralSimilarityIndexMeasure()

    def forward(self, image):
        return self.model(image)

    def shared_step(self, image, target):
        mul, add, prediction = self(image)
        loss = self.loss_fn(prediction, target)

        return loss, prediction

    def training_step(self, batch, batch_idx):
        image, target = batch["image"], batch["target"]
        loss, prediction = self.shared_step(image, target)

        self.log("train/loss", loss, on_step=True, on_epoch=False, prog_bar=False)
        self.train_psnr(prediction, target)
        self.log(
            "train/psnr", self.train_psnr, on_step=True, on_epoch=False, prog_bar=True
        )
        self.train_ssim(prediction, target)
        self.log(
            "train/ssim", self.train_ssim, on_step=True, on_epoch=False, prog_bar=True
        )

        return loss

    def validation_step(self, batch, batch_idx):
        image, target = batch["image"], batch["target"]
        loss, prediction = self.shared_step(image, target)

        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.val_psnr(prediction, target)
        self.log("val/psnr", self.val_psnr, on_step=False, on_epoch=True, prog_bar=True)
        self.val_ssim(prediction, target)
        self.log("val/ssim", self.val_ssim, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        mul, add, prediction = self(batch["image"])
        return prediction

    def configure_optimizers(self):
        return get_optimizers(self, self.config.optimizer)

    def on_train_start(self):
        self.logger.log_hyperparams(self.config)

        config_path = (
            Path(self.logger.experiment.project)
            / self.logger.experiment.id
            / "config.yaml"
        )
        config_path.parent.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(self.config, config_path)
