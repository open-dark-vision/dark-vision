from pathlib import Path
from typing import Dict, Optional

import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split

from src.datasets import SICE  # noqa: I900
from src.transforms.assemble_transform import create_transform  # noqa: I900


class SICEDataModule(pl.LightningDataModule):
    def __init__(self, config: Dict):
        super().__init__()
        self.data_path = Path(config["path"])
        self.batch_size = config["batch_size"]
        self.val_size = config["val_size"]

        self.max_exposure_ratio = config["max_exposure_ratio"]
        self.train_pair_selection_method = config["train_pair_selection_method"]
        self.test_pair_selection_method = config["test_pair_selection_method"]

        self.pin_memory = config["pin_memory"]
        self.num_workers = config["num_workers"]

        self.train_transform = (
            create_transform(**config["train_transform"])
            if "train_transform" in config
            else None
        )
        self.test_transform = (
            create_transform(**config["test_transform"])
            if "test_transform" in config
            else None
        )

    def setup(self, stage: Optional[str] = None):
        sice_full = SICE(
            self.data_path,
            train=True,
            pair_transform=self.train_transform,
            max_exposure_ratio=self.max_exposure_ratio,
            pair_selection_method=self.train_pair_selection_method,
        )

        self.sice_train, self.sice_val = random_split(
            sice_full, [1 - self.val_size, self.val_size]
        )
        self.sice_test = SICE(
            self.data_path,
            train=False,
            pair_transform=self.test_transform,
            max_exposure_ratio=self.max_exposure_ratio,
            pair_selection_method=self.test_pair_selection_method,
        )

    def train_dataloader(self):
        return DataLoader(
            self.sice_train,
            batch_size=self.batch_size,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.sice_val,
            batch_size=self.batch_size,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.sice_test,
            batch_size=self.batch_size,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
        )

    def predict_dataloader(self):
        return self.test_dataloader()
