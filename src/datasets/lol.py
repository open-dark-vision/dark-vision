from pathlib import Path
from typing import Dict, Optional

import albumentations as A
import pytorch_lightning as pl
from albumentations.pytorch.transforms import ToTensorV2
from torch.utils.data import DataLoader, Dataset, random_split

from src.datasets.meta import PairedImageInput  # noqa: I900
from src.transforms import load_transforms  # noqa: I900
from src.utils.image import read_image_cv2  # noqa: I900


class LOL(Dataset):
    def __init__(
        self,
        root: Path,
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
    def __init__(self, config: Dict):
        super().__init__()
        self.data_path = Path(config["path"])
        self.batch_size = config["batch_size"]
        self.val_size = config["val_size"]
        self.preload_dataset = config["preload"]

        self.pin_memory = config["pin_memory"]
        self.num_workers = config["num_workers"]

        self.train_transform, self.test_transform = load_transforms(config["transform"])

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
        return self.test_dataloader()
