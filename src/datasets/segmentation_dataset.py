# from datasets import Dataset, DatasetDict, Image
from pathlib import Path
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from src.configs.base import DataSegmentationConfig
import numpy as np
from transformers import SegformerFeatureExtractor


from torch.utils.data import Dataset
import os
from PIL import Image

class SemanticSegmentationDataset(Dataset):
    
    def __init__(self, config: DataSegmentationConfig):

        self.root = Path(config.path)
        self.config = config
        self.train = config.train
        self.feature_extractor = SegformerFeatureExtractor(reduce_labels=True)
        sub_path = "training" if self.train else "validation"
        sub_path = "testing" if self.config.test else sub_path
        sub_path = "predict" if self.config.predict else sub_path

        self.img_dir = os.path.join(self.root, "images", sub_path)
        self.ann_dir = os.path.join(self.root, "annotations", sub_path)

        self.prepare_images()
        self.prepare_annotations()

        self.loaded = self.config.preload
        if self.loaded:
            self.load_all()   

        assert len(self.images) == len(self.annotations), "There must be as many images as there are segmentation maps"

    def __len__(self):
        return len(self.images)
    
    def load_all(self):
        self.images_all = [Image.open(os.path.join(self.img_dir, img)) for img in self.images]
        self.annotations_all = [Image.open(os.path.join(self.ann_dir, ann)) for ann in self.annotations]

        encoded_inputs_all = [self.feature_extractor(img, ann, return_tensors="pt") for img, ann in zip(self.images_all, self.annotations_all)]

        for n, encoded_inputs in enumerate(encoded_inputs_all):
            for k,v in encoded_inputs.items():
                encoded_inputs[k].squeeze_()
            encoded_inputs_all[n] = encoded_inputs

        return encoded_inputs_all
    
    def prepare_images(self):
        image_file_names = []
        for root, dirs, files in os.walk(self.img_dir):
          image_file_names.extend(files)
        self.images = sorted(image_file_names)


    def prepare_annotations(self):
        annotation_file_names = []
        for root, dirs, files in os.walk(self.ann_dir):
          annotation_file_names.extend(files)
        self.annotations = sorted(annotation_file_names)
    
       
    def __getitem__(self, idx):
        
        image = Image.open(os.path.join(self.img_dir, self.images[idx]))
        segmentation_map = Image.open(os.path.join(self.ann_dir, self.annotations[idx]))

    

        id_to_trainid = {7: 1, 8: 2, 11: 3, 12: 4, 13: 5, 17: 6,
                              19: 7, 20: 8, 21: 9, 22: 10, 23: 11, 24: 12, 25: 13,
                              26: 14, 27: 15, 28: 16, 31: 17, 32: 18, 33: 19}
        
        label_copy = np.zeros(np.array(segmentation_map).shape, dtype=np.uint8)
        for k, v in id_to_trainid.items():
            label_copy[np.array(segmentation_map) == k] = v

        
        encoded_inputs = self.feature_extractor(image, label_copy, return_tensors="pt")

        for k,v in encoded_inputs.items():
          encoded_inputs[k].squeeze_() 
        
        return encoded_inputs



class SemanticDataModule(pl.LightningDataModule):
    def __init__(self, config: DataSegmentationConfig):
        super().__init__()
        self.config = config

    def setup(self, stage=None):
        self.train_dataset = SemanticSegmentationDataset(self.config)
        self.val_dataset = SemanticSegmentationDataset(self.config)

        if self.config.predict:
            self.predict_dataset = SemanticSegmentationDataset(self.config)

        if self.config.test:
            self.test_dataset = SemanticSegmentationDataset(self.config)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.config.batch_size, shuffle=True, num_workers=self.config.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.config.batch_size, shuffle=False, num_workers=self.config.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.config.batch_size, shuffle=False, num_workers=self.config.num_workers)

    def predict_dataloader(self):
        return DataLoader(self.predict_dataset, batch_size=self.config.batch_size, shuffle=False, num_workers=self.config.num_workers)

