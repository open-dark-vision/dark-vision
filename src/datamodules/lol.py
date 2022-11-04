from pathlib import Path
from typing import Callable, Dict, Optional

import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split

from src.datasets.lol import LOL  # noqa: I900


class LOLDataModule(pl.LightningDataModule):
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
        self.preload_dataset = config["preload"]

        self.pin_memory = config["pin_memory"]
        self.num_workers = config["num_workers"]

        self.train_transform = train_transform
        self.test_transform = test_transform

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: Optional[str] = None):
        LOL_full = LOL(
            self.data_path,
            train=True,
            pair_transform=self.train_transform,
            preload=self.preload_dataset,
        )

        self.LOL_train, self.LOL_val = random_split(
            LOL_full, [1 - self.val_size, self.val_size]
        )
        self.LOL_test = LOL(
            self.data_path,
            train=False,
            pair_transform=self.test_transform,
            preload=self.preload_dataset,
        )

    def train_dataloader(self):
        return DataLoader(
            self.LOL_train,
            batch_size=self.batch_size,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.LOL_val,
            batch_size=self.batch_size,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.LOL_test,
            batch_size=self.batch_size,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
        )

    def predict_dataloader(self):
        return self.test_transform()
