from transformers import SegformerForSemanticSegmentation
import json
from huggingface_hub import cached_download, hf_hub_url
import sys

from datasets import load_metric
from omegaconf import OmegaConf
import pytorch_lightning as pl
from model_segmentation import LitSegformerModel
from src.configs.experiments import segment_config as cfg
import os
from src.datasets import SemanticDataModule
import sys

from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
)

def create_json(dic_class):
    label2id = dic_class
    with open('id2label.json', 'w') as fp:
        json.dump(label2id, fp)


if __name__ == "__main__":

    if cfg.dataset.id2label_file is None:
        create_json(cfg.dataset.dic_class)

    cfg = OmegaConf.structured(cfg)

    
    seg_dm = SemanticDataModule(config=cfg.dataset)
   
    id2label = json.load(open(cfg.dataset.id2label_file, "r"))
    label2id = {v: k for k, v in id2label.items()}
    id2label = {v: k for k, v in label2id.items()}
    

    model = LitSegformerModel(cfg, label2id, id2label)
    callbacks = [
        RichProgressBar(),
        ModelCheckpoint(
            dirpath=os.path.dirname(cfg.checkpoint_path) if cfg.checkpoint_path is not None else None,
            monitor="val/loss",
            mode="min",
            save_top_k=2,
            save_last=True,
            auto_insert_metric_name=False,
            filename=cfg.name + "-" + cfg.model_name + "-{epoch:03d}-loss-{val/loss:.4f}",
        ),
    ]



    trainer = pl.Trainer(
        accelerator=cfg.device,
        devices=1,
        callbacks=callbacks,
        max_epochs=cfg.epochs,
    )

    trainer.fit(model, seg_dm)

