from pathlib import Path
from typing import Dict, Optional

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from src.datasets import Supplementary  # noqa: I900
from src.transforms.assemble_transform import create_transform  # noqa: I900


class SupplementaryDataModule(pl.LightningDataModule):
    def __init__(self, config: Dict):
        super().__init__()
        self.data_path = Path(config["path"])
        self.dataset_name = config["name"]
        self.batch_size = config["batch_size"]

        self.pin_memory = config["pin_memory"]
        self.num_workers = config["num_workers"]

        self.test_transform = (
            create_transform(**config["test_transform"])
            if "test_transform" in config
            else None
        )

    def setup(self, stage: Optional[str] = None):
        self.supplementary = Supplementary(
            self.data_path,
            self.dataset_name,
            transform=self.test_transform,
        )

    def train_dataloader(self):
        raise ValueError("Supplementary dataset has no train data")

    def test_dataloader(self):
        return DataLoader(
            self.supplementary,
            batch_size=self.batch_size,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
        )

    def predict_dataloader(self):
        return self.test_dataloader()
