from pathlib import Path

import pytorch_lightning as pl
import torch
import torchmetrics
from torchvision.utils import save_image

from src.models.LLFlow.conditional_encoder import ConditionalEncoder  # noqa: I900
from src.models.LLFlow.conditional_flow import ConditionalFlow  # noqa: I900
from src.models.LLFlow.flow_layers import SqueezeLayer  # noqa: I900
from src.models.LLFlow.utils import LLFlowNLL  # noqa: I900
from src.utils import get_optimizers  # noqa: I900


class LitLLFlow(pl.LightningModule):
    def __init__(self, config, save_images=3):
        super().__init__()
        self.config = config
        self.squeeze = SqueezeLayer(factor=8)
        self.save_images = save_images

        self.encoder = ConditionalEncoder(
            channels_in=config.model.encoder.channels_in,
            channels_middle=config.model.encoder.channels_middle,
            rrdb_number=config.model.encoder.rrdb_number,
            rrdb_channels=config.model.encoder.rrdb_channels,
        )

        self.flow = ConditionalFlow(
            n_levels=config.model.flow.n_levels,
            n_flow_steps=config.model.flow.n_flow_steps,
            n_additional_steps=config.model.flow.n_additional_steps,
            n_conditional_channels=config.model.flow.n_conditional_channels,
        )

        self.loss_fn = LLFlowNLL(p=config.loss.p)

        self.val_psnr = torchmetrics.PeakSignalNoiseRatio()
        self.val_ssim = torchmetrics.StructuralSimilarityIndexMeasure()

    def forward(self, image):
        rrdb_features = self.encoder(image)
        logdet = torch.zeros(image.size(0), device=image.device)
        return self.flow(
            z=self.squeeze(rrdb_features["color_map"])[0],
            conditional_features=rrdb_features,
            logdet=logdet,
            reverse=True,
        )

    def shared_step(self, image, target, generate_image=False):
        rrdb_features = self.encoder(image)

        z, logdet = self.flow(
            gt=target,
            conditional_features=rrdb_features,
            logdet=torch.zeros(image.size(0), device=image.device),
            reverse=False,
        )

        loss = self.loss_fn(z, logdet, rrdb_features["color_map"], target)
        prediction = (
            self.flow(
                z=self.squeeze(rrdb_features["color_map"])[0],
                conditional_features=rrdb_features,
                logdet=torch.zeros(image.size(0), device=image.device),
                reverse=True,
            )[0]
            if generate_image
            else None
        )

        return loss.mean(), prediction

    def training_step(self, batch, batch_idx):
        image, target = batch["image"], batch["target"]
        loss, _ = self.shared_step(image, target)

        self.log("train/loss", loss, on_step=True, on_epoch=False, prog_bar=False)

        return loss

    def validation_step(self, batch, batch_idx):
        image, target = batch["image"], batch["target"]
        loss, prediction = self.shared_step(image, target, generate_image=True)

        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.val_psnr(prediction, target)
        self.log("val/psnr", self.val_psnr, on_step=False, on_epoch=True, prog_bar=True)
        self.val_ssim(prediction, target)
        self.log("val/ssim", self.val_ssim, on_step=False, on_epoch=True, prog_bar=True)

        # save images
        if self.save_images > 0 and batch_idx == 0:
            for i in range(min(self.save_images, image.size(0))):
                Path(f"generated_imgs/{i}").mkdir(parents=True, exist_ok=True)
                save_image(
                    prediction[i], f"generated_imgs/{i}/{self.current_epoch}.jpg"
                )

        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        prediction, _ = self(batch["image"])
        return prediction

    def configure_optimizers(self):
        return get_optimizers(self, self.config.optimizer)
