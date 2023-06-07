
import torch
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torchmetrics
from omegaconf import OmegaConf
from torch import nn
from torchvision.utils import save_image

from src.models.KinD.DecomNet import DecomNet  # noqa: I900
from src.models.KinD.RestorationNet import RestorationNet  # noqa: I900
from src.models.KinD.IlluminationAdjustNet import IlluminationAdjustNet
from src.utils import get_loss, get_optimizers  # noqa: I900
from src.models.KinD.utils import KinDLoss_decom,  KinDLoss_restore, KinDLoss_illumina # noqa: I900




class LitKinD_decom(pl.LightningModule):
    def __init__(self, config, save_images=0):
        super().__init__()
        self.config = config
        self.save_images = save_images
        self.model = DecomNet()
        self.loss_fn = KinDLoss_decom()

        self.train_psnr = torchmetrics.PeakSignalNoiseRatio()
        self.train_ssim = torchmetrics.StructuralSimilarityIndexMeasure()

        self.val_psnr = torchmetrics.PeakSignalNoiseRatio()
        self.val_ssim = torchmetrics.StructuralSimilarityIndexMeasure()


    def forward(self, x):
        return self.model(x)
    
    
    def shared_step(self, image, target):
        reflect_1, illumin_1 = self.model(image) #low
        reflect_2, illumin_2 = self.model(target) #high
        loss = self.loss_fn(reflect_1, reflect_2, illumin_1, illumin_2, image, target, self.config.device)
        prediction = [reflect_1, illumin_1, reflect_2, illumin_2]
        return loss, prediction

    def training_step(self, batch, batch_idx):

        image, target = batch["image"], batch["target"]
        loss, prediction = self.shared_step(image, target)

        self.log("train/loss_decom", loss, on_step=True, on_epoch=False, prog_bar=False)


        return loss

    def validation_step(self, batch, batch_idx):
        image, target = batch["image"], batch["target"]
        loss, prediction = self.shared_step(image, target)

        self.log("val/loss_decom", loss, on_step=False, on_epoch=True, prog_bar=True)

        I_low = torch.cat([prediction[1],prediction[1],prediction[1]], axis=1)
        I_low = torch.permute(I_low, (0,1,3,2))
        I_high = torch.cat([prediction[3],prediction[3],prediction[3]], axis=1)
        I_high = torch.permute(I_high, (0,1,3,2))
        prediction[0] = torch.permute(prediction[0], (0,1,3,2))
        prediction[2] = torch.permute(prediction[2], (0,1,3,2))
        prediction[3] = I_high
        prediction[1] = I_low
        if self.save_images > 0 and batch_idx == 0:
            for i in range(min(self.save_images, image.size(0))):
                Path(f"generated_imgs_decom/{i}").mkdir(parents=True, exist_ok=True)
                save_image(
                    prediction[0][i], f"generated_imgs_decom/{i}/reflect_low.jpg"
                )
                save_image(
                    I_low[i], f"generated_imgs_decom/{i}/illum_low.jpg"
                )
                save_image(
                    prediction[2][i], f"generated_imgs_decom/{i}/reflect_high.jpg"
                )
                save_image(
                    I_high[i], f"generated_imgs_decom/{i}/illum_high.jpg"
                )


        return loss

    def configure_optimizers(self):
        return get_optimizers(self, self.config.optimizer)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        image, target = batch["image"], batch["target"]
        reflect_1, illumin_1 = self.model(image) #low
        reflect_2, illumin_2 = self.model(target) #high

        prediction = [reflect_1, illumin_1, reflect_2, illumin_2]

        I_low = torch.cat([illumin_1,illumin_1, illumin_1], axis=1)
        I_low = torch.permute(I_low, (0,1,3,2))
        I_high = torch.cat([prediction[3],prediction[3],prediction[3]], axis=1)
        I_high = torch.permute(I_high, (0,1,3,2))
        prediction[0] = torch.permute(prediction[0], (0,1,3,2))
        prediction[2] = torch.permute(prediction[2], (0,1,3,2))
        prediction[3] = I_high
        prediction[1] = I_low
        
        if self.save_images > 0 and batch_idx == 0:
            for i in range(min(self.save_images, image.size(0))):
                Path(f"generated_imgs_decom/{i}").mkdir(parents=True, exist_ok=True)
                save_image(
                    prediction[0][i], f"generated_imgs_decom/{i}/reflect_low.jpg"
                )
                save_image(
                    I_low[i], f"generated_imgs_decom/{i}/illum_low.jpg"
                )
                save_image(
                    prediction[2][i], f"generated_imgs_decom/{i}/reflect_high.jpg"
                )
                save_image(
                    I_high[i], f"generated_imgs_decom/{i}/illum_high.jpg"
                )
       
        return prediction
    

class LitKinD_restore(pl.LightningModule):
    def __init__(self, config, save_images=0):
        super().__init__()
        self.config = config
        self.save_images = save_images
        self.model = RestorationNet()
        self.loss_fn = KinDLoss_restore()
        
    def forward(self, x):
        return self.model(x)
    
    
    def shared_step(self, reflect_1, illumin_1, reflect_2):
        illumin_1 = torch.permute(illumin_1, (0,3,1,2))
        reflect_1 = torch.permute(reflect_1, (0,3,1,2))
        reflect_2 = torch.permute(reflect_2, (0,3,1,2))

        reflect_3 = self.model(reflect_1, illumin_1)
        loss = self.loss_fn(reflect_3, reflect_2, self.config.device) # reflect_2 is high image

        return loss, reflect_3
    

    def training_step(self, batch, batch_idx):

        reflect_1, illumin_1, reflect_2 = batch["reflect_low"], batch["illum_low"], batch["reflect_high"]

        loss, prediction = self.shared_step(reflect_1, illumin_1, reflect_2)

        self.log("train/loss_restore", loss, on_step=True, on_epoch=False, prog_bar=False)

        return loss

    def validation_step(self, batch, batch_idx):
        
        reflect_1, illumin_1, reflect_2 = batch["reflect_low"], batch["illum_low"], batch["reflect_high"]

        loss, prediction = self.shared_step(reflect_1, illumin_1, reflect_2)

        self.log("val/loss_restore", loss, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss

    def configure_optimizers(self):
        return get_optimizers(self, self.config.optimizer_restoration)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        reflect_1, illumin_1, reflect_2 = batch["reflect_low"], batch["illum_low"], batch["reflect_high"]
        reflect_3 = self.model(reflect_1, illumin_1)

        if self.save_images > 0 and batch_idx == 0:
            for i in range(min(self.save_images, illumin_1.size(0))):
                Path(f"generated_imgs_restore/{i}").mkdir(parents=True, exist_ok=True)
                save_image(
                    reflect_3, f"generated_imgs_restore/{i}/illumina.jpg"
                )
                
                
        return reflect_3
    
    


class LitKinD_illumina(pl.LightningModule):
    def __init__(self, config, save_images=0):
        super().__init__()
        self.config = config
        self.save_images = save_images
        self.model = IlluminationAdjustNet()
        self.loss_fn = KinDLoss_illumina()

        self.ratio = torch.Tensor([config.model.ratio])

    def forward(self, x):
        return self.model(x)
    
    def shared_step(self, illumin_1, illumin_2):
        
        illumin_2 = torch.permute(illumin_2, (0,3,1,2))
        illumin_1 = torch.permute(illumin_1, (0,3,1,2))

        illumin_3 = self.model(illumin_1, self.ratio)
        fusion = torch.cat((illumin_3, illumin_3, illumin_3), 1)
        
        
        loss = self.loss_fn(illumin_3, illumin_2, self.config.device)
        prediction = [illumin_3, fusion]
        return loss, prediction

    
    def training_step(self, batch, batch_idx):
            
        reflect_1, illumin_1, illumin_2 = batch["reflect_low"], batch["illum_low"],  batch["illum_high"] 
        loss, prediction = self.shared_step(illumin_1, illumin_2)
    
        self.log("train/loss_illumina", loss, on_step=True, on_epoch=False, prog_bar=False)
    
        return loss
    
    def validation_step(self, batch, batch_idx):
        reflect_1, illumin_1, illumin_2 = batch["reflect_low"], batch["illum_low"],  batch["illum_high"]

        loss, prediction = self.shared_step(illumin_1, illumin_2)
        
        self.log("val/loss_illumina", loss, on_step=False, on_epoch=True, prog_bar=True)


        if self.save_images > 0 and batch_idx == 0:
            for i in range(min(self.save_images, illumin_1.size(0))):
                Path(f"generated_imgs_illumina/{i}").mkdir(parents=True, exist_ok=True)
                save_image(
                    prediction[0][i], f"generated_imgs_illumina/{i}/illumina.jpg"
                )
                save_image(
                    prediction[1][i], f"generated_imgs_illumina/{i}/fusion.jpg"
                )
                
        return loss 
    
    
    def configure_optimizers(self):
        return get_optimizers(self, self.config.optimizer)
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        illumin_1, illumin_2 = batch["illum_low"],  batch["illum_high"]
        illumin_3 = self.model(illumin_1,self.ratio) 
        fusion = torch.cat((illumin_3, illumin_3, illumin_3), 1)

        prediction = [illumin_3, fusion]

        if self.save_images > 0 and batch_idx == 0:
            for i in range(min(self.save_images, illumin_1.size(0))):
                Path(f"generated_imgs_illumina/{i}").mkdir(parents=True, exist_ok=True)
                save_image(
                    prediction[0][i], f"generated_imgs_illumina/{i}/illumina.jpg"
                )
                save_image(
                    prediction[1][i], f"generated_imgs_illumina/{i}/fusion.jpg"
                )
                
        return prediction
    

class LitKinD(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.decom_net = DecomNet()
        self.restore_net = RestorationNet()
        self.illum_net = IlluminationAdjustNet()
        self.ratio = torch.Tensor([config.model.ratio])

        # self.psnr = torchmetrics.PeakSignalNoiseRatio()
        # self.ssim = torchmetrics.StructuralSimilarityIndexMeasure()

    
    def forward(self, image):
        reflect_1, illumin_1 = self.decom_net(image) # low light image
        reflect_1 = torch.permute(reflect_1, (0,1,3,2))
        illumin_1 = torch.permute(illumin_1, (0,1,3,2))
        illumin_3 = self.illum_net(illumin_1, self.ratio)
        reflect_finale = self.restore_net(reflect_1, illumin_1)
        illumin_finale = torch.cat((illumin_3, illumin_3, illumin_3), 1)
        output = reflect_finale * illumin_finale

        return reflect_finale, illumin_finale, output
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        image, target = batch["image"], batch["target"]
        reflect_finale, illumin_finale, output= self(image)
        target = torch.permute(target, (0,3,2,1))
       
        # self.psnr(output, target)
        # self.log("metric/psnr", self.psnr, on_step=False, on_epoch=True, prog_bar=True)
        # self.ssim(output, target)
        # self.log("metric/ssim", self.ssim, on_step=False, on_epoch=True, prog_bar=True)

        return output



