from pathlib import Path

import pytorch_lightning as pl
import torch
import torchmetrics
from omegaconf import OmegaConf
from torch import nn

from src.models.SCI.calibrate_net import CalibrateNetwork  # noqa: I900
from src.models.SCI.enhance_net import EnhanceNetwork  # noqa: I900
from src.utils import get_loss, get_optimizers  # noqa: I900

# based on https://github.com/vis-opt-group/SCI


class SCI(nn.Module):
    def __init__(self, stage: int = 3, finetune: bool = False, weights: str = None):
        super().__init__()
        self.stage = stage
        self.finetune = finetune

        self.enhance = EnhanceNetwork(layers=1, channels=3)
        self.enhance.in_conv.apply(SCI.weights_init)
        self.enhance.conv.apply(SCI.weights_init)
        self.enhance.out_conv.apply(SCI.weights_init)

        if not finetune:
            self.calibrate = CalibrateNetwork(
                layers=3, channels=16
            )  # TODO: does layers=3 imply 3 stages?
            self.calibrate.in_conv.apply(SCI.weights_init)
            self.calibrate.convs.apply(SCI.weights_init)
            self.calibrate.out_conv.apply(SCI.weights_init)

        if weights is not None:
            print(f"Loading weights from {weights}...")
            self.load_weights(weights)

    def load_weights(self, weights):
        base_weights = torch.load(weights)
        pretrained_dict = base_weights

        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)

        self.load_state_dict(model_dict)

    @staticmethod
    def weights_init(m):
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()

        if isinstance(m, nn.BatchNorm2d):
            m.weight.data.normal_(1.0, 0.02)

    def forward(self, images):
        if self.finetune:
            illumination = self.enhance(images)
            reflectance = images / illumination
            reflectance = torch.clamp(reflectance, 0, 1)
            return illumination, reflectance
        else:
            return self.forward_calibrate(images)

    def forward_calibrate(self, images):
        ilist, rlist, inlist, attlist = [], [], [], []
        input_op = images

        for _ in range(self.stage):
            inlist.append(input_op)
            illumination = self.enhance(input_op)

            reflectance = images / illumination
            reflectance = torch.clamp(reflectance, 0, 1)

            att = self.calibrate(reflectance)
            input_op = images + att

            ilist.append(illumination)
            rlist.append(reflectance)
            attlist.append(torch.abs(att))

        return ilist, rlist, inlist, attlist


class LitSCI(pl.LightningModule):
    def __init__(self, config, weights=None):
        super().__init__()
        self.config = config
        self.finetune = config.finetune
        self.supervised_metrics = config.model.supervised_metrics

        self.model = SCI(
            stage=config.model.stage, finetune=self.finetune, weights=weights
        )
        self.loss_fn = get_loss(config.loss, finetune=self.finetune)

        if self.supervised_metrics:
            self.train_psnr = torchmetrics.PeakSignalNoiseRatio()
            self.train_ssim = torchmetrics.StructuralSimilarityIndexMeasure()

            self.val_psnr = torchmetrics.PeakSignalNoiseRatio()
            self.val_ssim = torchmetrics.StructuralSimilarityIndexMeasure()

    def forward(self, images):
        return self.model(images)

    def shared_step(self, batch):
        images = batch["image"]

        if self.finetune:
            illumination, reflectance = self(images)
            loss = self.loss_fn(images, illumination)
        else:  # calibrate, returns lists
            illumination, reflectance, inputs, _ = self(images)
            reflectance = reflectance[0]
            loss = self.loss_fn(inputs, illumination)

        return loss, illumination, reflectance

    def training_step(self, batch, batch_idx):
        loss, illumination, reflectance = self.shared_step(batch)

        self.log("train/loss", loss, on_step=True, on_epoch=False, prog_bar=False)

        if self.supervised_metrics:
            target = batch["target"]

            self.train_psnr(reflectance, target)
            self.log(
                "train/psnr",
                self.train_psnr,
                on_step=True,
                on_epoch=False,
                prog_bar=True,
            )
            self.train_ssim(reflectance, target)
            self.log(
                "train/ssim",
                self.train_ssim,
                on_step=True,
                on_epoch=False,
                prog_bar=True,
            )
        return loss

    def validation_step(self, batch, batch_idx):
        loss, illumination, reflectance = self.shared_step(batch)

        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        if self.supervised_metrics:
            target = batch["target"]

            self.val_psnr(reflectance, target)
            self.log(
                "val/psnr", self.val_psnr, on_step=False, on_epoch=True, prog_bar=True
            )
            self.val_ssim(reflectance, target)
            self.log(
                "val/ssim", self.val_ssim, on_step=False, on_epoch=True, prog_bar=True
            )
        return loss

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
