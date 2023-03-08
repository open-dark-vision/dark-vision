from pathlib import Path

import pytorch_lightning as pl
import torchmetrics
from torchvision.utils import save_image

from src.models.BriDiMo.unet import ConditionalUNet  # noqa: I900
from src.utils import get_loss, get_optimizers  # noqa: I900


class LitBriDiMo(pl.LightningModule):
    def __init__(self, config, save_images=3):
        super().__init__()
        self.config = config
        self.save_images = save_images

        self.backbone = ConditionalUNet(
            n_channels=config.model.in_channels, bilinear=config.model.bilinear
        )

        self.loss_fn = get_loss(config.loss)

        self.val_psnr = torchmetrics.PeakSignalNoiseRatio()
        self.val_ssim = torchmetrics.StructuralSimilarityIndexMeasure()

    def forward(self, image, source_lightness, target_lightness):
        return self.backbone(image, source_lightness, target_lightness)

    def shared_step(self, batch):
        image, target = batch["image"], batch["target"]
        source_lightness = batch["source_lightness"]
        target_lightness = batch["target_lightness"]

        predicted = self.forward(image, source_lightness, target_lightness)

        loss = self.loss_fn(target, predicted)

        return loss.mean(), predicted

    def training_step(self, batch, batch_idx):
        loss, _ = self.shared_step(batch)

        self.log("train/loss", loss, on_step=True, on_epoch=False, prog_bar=False)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, predictions = self.shared_step(batch)

        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.val_psnr(predictions, batch["target"])
        self.log("val/psnr", self.val_psnr, on_step=False, on_epoch=True, prog_bar=True)
        self.val_ssim(predictions, batch["target"])
        self.log("val/ssim", self.val_ssim, on_step=False, on_epoch=True, prog_bar=True)

        # save images
        if self.save_images > 0 and batch_idx == 0:
            for i in range(min(self.save_images, predictions.size(0))):
                Path("BriDiMo_imgs/originals/").mkdir(parents=True, exist_ok=True)
                Path(f"BriDiMo_imgs/dimmed/{i}").mkdir(parents=True, exist_ok=True)
                Path(f"BriDiMo_imgs/histogram/{i}").mkdir(parents=True, exist_ok=True)
                Path(f"BriDiMo_imgs/color_map/{i}").mkdir(parents=True, exist_ok=True)
                Path(f"BriDiMo_imgs/generated/{i}").mkdir(parents=True, exist_ok=True)
                save_image(
                    predictions[i],
                    f"BriDiMo_imgs/generated/{i}/{self.current_epoch}.jpg",
                )
                save_image(
                    batch["image"][i, :3, :, :],
                    f"BriDiMo_imgs/dimmed/{i}/{self.current_epoch}.jpg",
                )
                save_image(
                    batch["image"][i, 3:6, :, :],
                    f"BriDiMo_imgs/histogram/{i}/{self.current_epoch}.jpg",
                )
                save_image(
                    batch["image"][i, 6:9, :, :],
                    f"BriDiMo_imgs/color_map/{i}/{self.current_epoch}.jpg",
                )
                save_image(
                    batch["target"][i], f"BriDiMo_imgs/originals/{i}.jpg"
                )  # will be the same through all epochs

        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        prediction, _ = self(
            batch["image"], batch["source_lightness"], batch["target_lightness"]
        )
        return prediction

    def configure_optimizers(self):
        return get_optimizers(self, self.config.optimizer)
