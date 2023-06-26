import torch
import pytorch_lightning as pl
from transformers import SegformerForSemanticSegmentation
from datasets import load_metric
import numpy as np
from torch import nn

class LitSegformerModel(pl.LightningModule):
    def __init__(self, config, label2id, id2label):
        super().__init__()
        self.config = config
        self.model = SegformerForSemanticSegmentation.from_pretrained(self.config.model_name,
                                                            num_labels=len(label2id.items()), 
                                                            id2label=id2label, 
                                                            label2id=label2id,
    )
        self.metric = load_metric("mean_iou")
        self.label2id = label2id

    
    def forward(self, pixel_values, labels):
        outputs = self.model(pixel_values=pixel_values, labels=labels)
        return outputs
        
    def common_step(self, batch, batch_idx):
        pixel_values = batch['pixel_values']
        labels = batch['labels']
        
        outputs = self(pixel_values=pixel_values, labels=labels)
        logits = outputs.logits
        loss = outputs.loss
       

        upsampled_logits = nn.functional.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
        predicted = upsampled_logits.argmax(dim=1)

        self.metric.add_batch(predictions=predicted.detach().cpu().numpy(), references=labels.detach().cpu().numpy())

        return loss
      
    def training_step(self, batch, batch_idx):
        loss = self.common_step(batch, batch_idx)     

        self.log("training_loss", loss)
        # self.log("train_iou", self.metric._compute(num_labels=len(self.label2id.items()), 
        #                            ignore_index=255,
        #                            reduce_labels=False), on_epoch=True)


        return loss
    
    def validation_step(self, batch, batch_idx):
        loss = self.common_step(batch, batch_idx)     
        self.log("validation_loss", loss, on_epoch=True)
        # self.log("val_iou", self.metric._compute(num_labels=len(self.label2id.items()), ignore_index=255, reduce_labels=False), on_epoch=True)

        return loss

    def test_step(self, batch, batch_idx):
        loss = self.common_step(batch, batch_idx)     

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=0.00006)
        return optimizer