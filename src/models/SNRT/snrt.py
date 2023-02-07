import functools
from pathlib import Path  # noqa: F401

import kornia
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from einops import rearrange, reduce, repeat
from omegaconf import OmegaConf  # noqa: F401

from src.models.SNRT import arch_util  # noqa: I900
from src.models.SNRT.loss import CharbonnierLoss, VGGLoss  # noqa: I900
from src.models.SNRT.transformer import Encoder_patch66  # noqa: I900
from src.utils import get_optimizers  # noqa: I900


###############################
# https://github.com/dvlab-research/SNR-Aware-Low-Light-Enhance/tree/1113144c82adc8bcc4a9ec27749ed75f196a4e4d
class SNRT(nn.Module):
    def __init__(
        self,
        nf=64,
        nframes=5,
        front_RBs=5,
        back_RBs=10,
        center=None,
        predeblur=False,
        w_TSA=True,
    ):
        super().__init__()
        self.nf = nf
        self.center = nframes // 2 if center is None else center
        self.is_predeblur = True if predeblur else False
        self.w_TSA = w_TSA
        ResidualBlock_noBN_f = functools.partial(arch_util.ResidualBlock_noBN, nf=nf)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        self.conv_first_1 = nn.Conv2d(3, nf, 3, 1, 1, bias=True)
        self.conv_first_2 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
        self.conv_first_3 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)

        self.feature_extraction = arch_util.make_layer(ResidualBlock_noBN_f, front_RBs)
        self.recon_trunk = arch_util.make_layer(ResidualBlock_noBN_f, back_RBs)

        self.upconv1 = nn.Conv2d(nf * 2, nf * 4, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(nf * 2, 64 * 4, 3, 1, 1, bias=True)
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.HRconv = nn.Conv2d(64 * 2, 64, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1, bias=True)

        self.transformer = Encoder_patch66(d_model=1024, d_inner=2048, n_layers=6)
        self.recon_trunk_light = arch_util.make_layer(ResidualBlock_noBN_f, 6)

    def forward(self, x, mask=None):
        x_center = x

        L1_fea_1 = self.lrelu(self.conv_first_1(x_center))
        L1_fea_2 = self.lrelu(self.conv_first_2(L1_fea_1))
        L1_fea_3 = self.lrelu(self.conv_first_3(L1_fea_2))

        fea = self.feature_extraction(L1_fea_3)
        fea_light = self.recon_trunk_light(fea)

        h_feature = fea.shape[2]
        w_feature = fea.shape[3]
        mask = F.interpolate(mask, size=[h_feature, w_feature], mode="nearest")

        xs = np.linspace(-1, 1, fea.size(3) // 4)
        ys = np.linspace(-1, 1, fea.size(2) // 4)
        xs = np.meshgrid(xs, ys)
        xs = np.stack(xs, 2)
        xs = repeat(torch.Tensor(xs), "h w c -> b h w c", b=fea.size(0))
        xs = rearrange(xs, "b h w c -> b (h w) c")

        height = fea.shape[2]
        width = fea.shape[3]
        fea_unfold = F.unfold(fea, kernel_size=4, dilation=1, stride=4, padding=0)
        fea_unfold = rearrange(fea_unfold, "b h w -> b w h")

        mask_unfold = F.unfold(mask, kernel_size=4, dilation=1, stride=4, padding=0)
        mask_unfold = rearrange(mask_unfold, "b h w -> b w h")
        mask_unfold = reduce(mask_unfold, "b w h -> b 1 w", "mean")
        mask_unfold[mask_unfold <= 0.5] = 0.0

        fea_unfold = self.transformer(fea_unfold, xs, src_mask=mask_unfold)
        fea_unfold = rearrange(fea_unfold, "b w h -> b h w")
        fea_unfold = nn.Fold(
            output_size=(height, width),
            kernel_size=(4, 4),
            stride=4,
            padding=0,
            dilation=1,
        )(fea_unfold)

        channel = fea.shape[1]
        mask = repeat(mask, "b 1 h w -> b c h w", c=channel)
        fea = fea_unfold * (1 - mask) + fea_light * mask

        out_noise = self.recon_trunk(fea)
        out_noise = torch.cat([out_noise, L1_fea_3], dim=1)
        out_noise = self.lrelu(self.pixel_shuffle(self.upconv1(out_noise)))
        out_noise = torch.cat([out_noise, L1_fea_2], dim=1)
        out_noise = self.lrelu(self.pixel_shuffle(self.upconv2(out_noise)))
        out_noise = torch.cat([out_noise, L1_fea_1], dim=1)
        out_noise = self.lrelu(self.HRconv(out_noise))
        out_noise = self.conv_last(out_noise)
        out_noise = out_noise + x_center

        return out_noise


class LitSNRT(pl.LightningModule):
    def __init__(self, config=None):
        super().__init__()
        self.config = config

        self.model = SNRT(
            nf=config.model.nf,
            nframes=config.model.nframes,
            front_RBs=config.model.front_RBs,
            back_RBs=config.model.back_RBs,
            center=config.model.center,
            predeblur=config.model.predeblur,
            w_TSA=config.model.w_TSA,
        )

        self.loss_ch = CharbonnierLoss()
        self.lambd = config.model.lambd
        self.loss_vgg = VGGLoss()

        self.l_pix_w = config.model.l_pix_w

        blur_kernel = config.model.blur_kernel
        self.blur = kornia.filters.MedianBlur((blur_kernel, blur_kernel))

        self.train_psnr = torchmetrics.PeakSignalNoiseRatio()
        self.train_ssim = torchmetrics.StructuralSimilarityIndexMeasure()

        self.val_psnr = torchmetrics.PeakSignalNoiseRatio()
        self.val_ssim = torchmetrics.StructuralSimilarityIndexMeasure()

    def forward(self, x, mask):
        return self.model(x, mask)

    @staticmethod
    def weight_image(image):
        weights = [0.299, 0.587, 0.114]
        image = (
            image[:, 0:1, :, :] * weights[0]
            + image[:, 1:2, :, :] * weights[1]
            + image[:, 2:3, :, :] * weights[2]
        )
        return image

    def shared_step(self, image, target):
        dark = image.clone()
        light = self.blur(dark)

        dark = self.weight_image(dark)
        light = self.weight_image(light)

        noise = torch.abs(dark - light)
        mask = torch.div(light, noise + 0.0001)

        batch_size = mask.shape[0]
        height = mask.shape[2]
        width = mask.shape[3]
        mask_max = torch.max(mask.view(batch_size, -1), dim=1)[0]
        mask_max = rearrange(mask_max, "b -> b 1 1 1")
        mask_max = repeat(mask_max, "b 1 1 1 -> b 1 h w", h=height, w=width)
        mask = mask * 1.0 / (mask_max + 0.0001)

        mask = torch.clamp(mask, min=0, max=1.0)
        mask = mask.float()

        prediction = self(image, mask)

        l_pix = self.l_pix_w * self.loss_ch(prediction, target)
        l_vgg = self.l_pix_w * self.loss_vgg(prediction, target) * self.lambd
        loss = l_pix + l_vgg

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

        self.log("val/loss", loss, on_step=True, on_epoch=False, prog_bar=False)
        self.val_psnr(prediction, target)
        self.log("val/psnr", self.val_psnr, on_step=True, on_epoch=False, prog_bar=True)
        self.val_ssim(prediction, target)
        self.log("val/ssim", self.val_ssim, on_step=True, on_epoch=False, prog_bar=True)
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        prediction = self(batch["image"])
        return prediction

    def configure_optimizers(self):
        return get_optimizers(self, self.config.optimizer)

    # def on_train_start(self):
    #     self.logger.log_hyperparams(self.config)

    #     config_path = (
    #         Path(self.logger.experiment.project)
    #         / self.logger.experiment.id
    #         / "config.yaml"
    #     )
    #     config_path.parent.mkdir(parents=True, exist_ok=True)
    #     OmegaConf.save(self.config, config_path)
