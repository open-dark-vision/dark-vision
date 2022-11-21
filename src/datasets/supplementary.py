from pathlib import Path
from typing import Optional

import pytorch_lightning as pl
from albumentations.pytorch.transforms import ToTensorV2
from torch.utils.data import DataLoader, Dataset

from src.configs.base import (  # noqa: I900
    SupplementaryDataset,
    SupplementaryDatasetConfig,
)
from src.datasets.meta import Image  # noqa: I900
from src.transforms import load_transforms  # noqa: I900
from src.utils.image import read_image_cv2  # noqa: I900


class Supplementary(Dataset):
    def __init__(self, root: Path, dataset: SupplementaryDataset, transform=None):
        self.root = self.select_dataset(root, dataset)
        self.images_paths = list(
            self.root.glob("**/*.[JPG PNG jpg png JPEG jpeg bmp]*")
        )

        self.transform = transform if transform is not None else ToTensorV2()

    def select_dataset(self, root: Path, dataset_name: str) -> Path:
        for dataset in SupplementaryDataset:
            if dataset == dataset_name:
                return root / dataset.name

        raise ValueError(f"Dataset {dataset_name} not found in Supplementary datasets")

    def __len__(self) -> int:
        return len(self.images_paths)

    def __getitem__(self, index: int) -> Image:
        image_path = self.images_paths[index]
        image = read_image_cv2(image_path)
        image = self.transform(image=image)["image"]

        return {"image": image}


class SupplementaryDataModule(pl.LightningDataModule):
    def __init__(self, config: SupplementaryDatasetConfig):
        super().__init__()
        self.root = Path(config.path)
        self.config = config

        _, self.test_transform = load_transforms(config.transform)

    def setup(self, stage: Optional[str] = None):
        self.test_ds = Supplementary(
            self.root,
            self.config.name,
            transform=self.test_transform,
        )

    def train_dataloader(self):
        raise ValueError("Supplementary dataset has no train data")

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.config.batch_size,
            pin_memory=self.config.pin_memory,
            num_workers=self.config.num_workers,
        )

    def predict_dataloader(self):
        return self.test_dataloader()
