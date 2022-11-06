from pathlib import Path
from typing import Callable, Dict, Optional

import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split

from src.datasets import ExDark  # noqa: I900


class ExDarkDataModule(pl.LightningDataModule):
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

        self.pin_memory = config["pin_memory"]
        self.num_workers = config["num_workers"]

        self.train_transform = train_transform
        self.test_transform = test_transform

    def setup(self, stage: Optional[str] = None):
        exdark_full = ExDark(
            self.data_path,
            train=True,
            transform=self.train_transform,
        )

        self.exdark_train, self.exdark_val = random_split(
            exdark_full, [1 - self.val_size, self.val_size]
        )
        self.exdark_test = ExDark(
            self.data_path,
            train=False,
            transform=self.test_transform,
        )

    def train_dataloader(self):
        return DataLoader(
            self.exdark_train,
            batch_size=self.batch_size,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
            collate_fn=ExDark.collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.exdark_val,
            batch_size=self.batch_size,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
            collate_fn=ExDark.collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.exdark_test,
            batch_size=self.batch_size,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
            collate_fn=ExDark.collate_fn,
        )

    def predict_dataloader(self):
        return self.test_dataloader()
