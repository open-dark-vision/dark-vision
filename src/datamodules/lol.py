from pathlib import Path
from typing import Callable, Optional, Union

import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split

from src.datasets.lol import LOL  # noqa: I900


class LOLDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_path: Union[Path, str] = "data/LOL-fair",
        batch_size: int = 32,
        train_transform: Optional[Callable] = None,
        test_transform: Optional[Callable] = None,
        val_size: float = 0.05,
        preload_dataset: bool = False,
    ):
        super().__init__()
        self.data_path = Path(data_path) if isinstance(data_path, str) else data_path
        self.batch_size = batch_size
        self.val_size = val_size
        self.preload_dataset = preload_dataset

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
        self.LOL_predict = LOL(
            self.data_path,
            train=False,
            pair_transform=self.test_transform,
            preload=self.preload_dataset,
        )

    def train_dataloader(self):
        return DataLoader(self.LOL_train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.LOL_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.LOL_test, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.LOL_predict, batch_size=self.batch_size)


if __name__ == "__main__":
    lol_dm = LOLDataModule()
    lol_dm.setup()
    first_batch = next(iter(lol_dm.train_dataloader()))
    print(first_batch["image"].size())
    print(first_batch["target"].size())
