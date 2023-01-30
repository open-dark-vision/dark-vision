
import torch
from pathlib import Path
from DecomNet_simple import DecomNet
from Illumination_adjust_net import Illumination_adjust_net
from Restoration_net import Restoration_net

import pytorch_lightning as pl
import torchmetrics
from omegaconf import OmegaConf
from torch import nn

from src.models.KinD.DecomNet_simple import DecomNet  # noqa: I900
from src.models.KinD.Restoration_net import Restoration_net  # noqa: I900
from src.models.KinD.Illumination_adjust_net import Illumination_adjust_net
from src.utils import get_loss, get_optimizers  # noqa: I900

class KinD(nn.Module):
    def __init__(self):
        super().__init__()
        self.decom_net = DecomNet()
        self.restore_net = Restoration_net()
        self.illumina_net = Illumination_adjust_net()

    def forward(self, x, ratio):
        reflect_1, illumin_1 = self.decom_net(x)
        reflect_2 = self.restore_net(reflect_1, illumin_1)
        illumin_2 = self.illumina_net(illumin_1, ratio)
        fusion = torch.cat((illumin_2, illumin_2, illumin_2), 1)
        return reflect_2, illumin_2, fusion * reflect_2


class LitKinD(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = KinD()
        self.ratio = torch.tensor([0.5])
        self.loss_fn = get_loss(config.loss)

        self.train_psnr = torchmetrics.PeakSignalNoiseRatio()
        self.train_ssim = torchmetrics.StructuralSimilarityIndexMeasure()

        self.val_psnr = torchmetrics.PeakSignalNoiseRatio()
        self.val_ssim = torchmetrics.StructuralSimilarityIndexMeasure()

    def forward(self, x, ratio):
        return self.model(x, ratio)

    def shared_step(self, image, target):
        reflect, illumin, prediction = self(image, self.ratio)
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

    def configure_optimizers(self):
        return get_optimizers(self, self.config.optimizer)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        reflect, illumin, prediction = self(batch["image"])
        return prediction

    def on_train_start(self):
        self.logger.log_hyperparams(self.config)

        config_path = (
                Path(self.logger.experiment.project)
                / self.logger.experiment.id
                / "config.yaml"
        )
        config_path.parent.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(self.config, config_path)

