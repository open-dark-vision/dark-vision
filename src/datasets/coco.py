from pathlib import Path
from typing import List, Optional

import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset, random_split

from src.configs.base import COCODatasetConfig  # noqa: I900
from src.datasets.meta import PairedImageWithLightnessInput  # noqa: I900
from src.transforms import load_transforms, MCBFSTransform  # noqa: I900
from src.utils.image import read_image_cv2  # noqa: I900


class COCO(Dataset):
    def __init__(
        self,
        root: Path,
        transform: MCBFSTransform,
        indices: Optional[List[int]] = None,
    ):
        self.transform = transform

        self.image_names = sorted(root.glob("*.jpg"), key=lambda x: int(x.stem))

        if indices is not None:
            self.image_names = [self.image_names[index] for index in indices]

    def __len__(self) -> int:
        return len(self.image_names)

    def __getitem__(self, index: int) -> PairedImageWithLightnessInput:
        image = read_image_cv2(self.image_names[index])

        transformed = self.transform(light=image)
        image, target = transformed["image"], transformed["target"]

        source_lightness = transformed['source_lightness']
        target_lightness = transformed['target_lightness']

        return PairedImageWithLightnessInput(
            image=image,
            target=target,
            source_lightness=source_lightness,
            target_lightness=target_lightness
        )


class COCODataModule(pl.LightningDataModule):
    def __init__(self, config: COCODatasetConfig):
        super().__init__()
        self.root = Path(config.path)
        self.config = config

        self.train_transform, self.test_transform = load_transforms(config.transform)

    def setup(self, stage: Optional[str] = None):
        n_train_images = len(list(self.root.glob("*.jpg")))

        train_indices, val_indices = random_split(
            range(n_train_images), [1 - self.config.val_size, self.config.val_size]
        )

        self.train_ds = COCO(
            self.root,
            transform=self.train_transform,
            indices=train_indices,
        )

        self.val_ds = COCO(
            self.root,
            transform=self.test_transform,
            indices=val_indices,
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
        return self.val_dataloader()

    def predict_dataloader(self):
        return self.test_dataloader()
