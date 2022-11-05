from pathlib import Path
from typing import Callable, Dict, Optional

import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split

from src.datasets import SICE  # noqa: I900


class SICEDataModule(pl.LightningDataModule):
    def __init__(
        self,
        config: Dict,
        train_transform: Optional[Callable] = None,
        test_transform: Optional[Callable] = None,
    ):
        super().__init__()
        self.data_path = Path(config["path"])
        self.batch_size = config["batch_size"]
        self.val_size = config["val_size"]

        self.max_exposure_ratio = config["max_exposure_ratio"]
        self.train_pair_selection_method = config["train_pair_selection_method"]
        self.test_pair_selection_method = config["test_pair_selection_method"]

        self.pin_memory = config["pin_memory"]
        self.num_workers = config["num_workers"]

        self.train_transform = train_transform
        self.test_transform = test_transform

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
