from pathlib import Path
from typing import List, Optional

import albumentations as A
import pytorch_lightning as pl
from albumentations.pytorch.transforms import ToTensorV2
from torch.utils.data import DataLoader, Dataset, random_split

from src.configs.base import LOLDatasetConfig  # noqa: I900
from src.datasets.meta import PairedImageInput  # noqa: I900
from src.transforms import load_transforms  # noqa: I900
from src.utils.image import read_image_cv2  # noqa: I900


class LOL(Dataset):
    def __init__(
        self,
        root: Path,
        indices: Optional[List[int]] = None,
        train: bool = True,
        pair_transform: Optional[A.Compose] = None,
        preload: bool = False,
    ):
        self.pair_transform = (
            pair_transform
            if pair_transform is not None
            else A.Compose(
                [ToTensorV2()],
                additional_targets={"target": "image"},
            )
        )

        path = root / ("our485" if train else "eval15")

        self.image_names = sorted(
            (path / "low/").glob("*.png"), key=lambda x: int(x.stem)
        )
        self.target_names = sorted(
            (path / "high/").glob("*.png"), key=lambda x: int(x.stem)
        )

        if indices is not None:
            self.image_names = [self.image_names[index] for index in indices]
            self.target_names = [self.target_names[index] for index in indices]

        self.loaded = preload
        if self.loaded:
            self.load_all_images()

    def load_all_images(self) -> None:
        self.loaded_images_ = []
        self.loaded_targets_ = []

        for index in range(len(self)):
            self.loaded_images_.append(read_image_cv2(self.image_names[index]))
            self.loaded_targets_.append(read_image_cv2(self.image_names[index]))

    def __len__(self) -> int:
        return len(self.image_names)

    def __getitem__(self, index: int) -> PairedImageInput:
        image = (
            self.loaded_images_[index]
            if self.loaded
            else read_image_cv2(self.image_names[index])
        )
        target = (
            self.loaded_targets_[index]
            if self.loaded
            else read_image_cv2(self.target_names[index])
        )

        transformed = self.pair_transform(image=image, target=target)
        image, target = transformed["image"], transformed["target"]

        return PairedImageInput(image=image, target=target)


class LOLDataModule(pl.LightningDataModule):
    def __init__(self, config: LOLDatasetConfig):
        super().__init__()
        self.root = Path(config.path)
        self.config = config

        self.train_transform, self.test_transform = load_transforms(config.transform)

    def setup(self, stage: Optional[str] = None):
        n_train_images = len(list((self.root / "our485/low/").glob("*.png")))
        train_indices, val_indices = random_split(
            range(n_train_images), [1 - self.config.val_size, self.config.val_size]
        )

        self.train_ds = LOL(
            self.root,
            indices=train_indices,
            train=True,
            pair_transform=self.train_transform,
            preload=self.config.preload,
        )

        self.val_ds = LOL(
            self.root,
            indices=val_indices,
            train=True,
            pair_transform=self.test_transform,
            preload=self.config.preload,
        )

        self.test_ds = LOL(
            self.root,
            train=False,
            pair_transform=self.test_transform,
            preload=self.config.preload,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.config.batch_size,
            pin_memory=self.config.pin_memory,
            num_workers=self.config.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.config.batch_size,
            pin_memory=self.config.pin_memory,
            num_workers=self.config.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.config.batch_size,
            pin_memory=self.config.pin_memory,
            num_workers=self.config.num_workers,
        )

    def predict_dataloader(self):
        return self.test_dataloader()
