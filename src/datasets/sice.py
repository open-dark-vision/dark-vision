import random
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import albumentations as A
import numpy as np
import pytorch_lightning as pl
from albumentations.pytorch.transforms import ToTensorV2
from torch.utils.data import DataLoader, Dataset, random_split

from src.configs.base import PairSelectionMethod, SICEDatasetConfig  # noqa: I900
from src.datasets.meta import PairedImageInput  # noqa: I900
from src.transforms import load_transforms  # noqa: I900
from src.utils.image import read_image_cv2  # noqa: I900


class PairSelector:
    def __init__(self, method: PairSelectionMethod, max_exposure_ratio: float = 1.0):
        self.method = method
        self.max_exposure_ratio = max_exposure_ratio

    def sample_under_exposure(self, sequence: List[Path]) -> Tuple[Path, Path]:
        n_images = len(sequence)
        percentages = (np.arange(n_images) + 1) / n_images

        n_images_under_exposure = (percentages <= self.max_exposure_ratio).sum()
        return sequence[:n_images_under_exposure]

    def random_next(self, sequence: List[Path]) -> Tuple[Path, Path]:
        if self.max_exposure_ratio < 1.0:
            sequence = self.sample_under_exposure(sequence)

        idx = random.randint(0, len(sequence) - 2)
        return sequence[idx], sequence[idx + 1]

    def random_target(self, sequence: List[Path], target: Path) -> Tuple[Path, Path]:
        if self.max_exposure_ratio < 1.0:
            sequence = self.sample_under_exposure(sequence)

        idx = random.randint(0, len(sequence) - 1)
        return sequence[idx], target

    def halfexp_target(self, sequence: List[Path], target: Path) -> Tuple[Path, Path]:
        idx = len(sequence) // 2
        return sequence[idx], target

    def random_halfexp(self, sequence: List[Path]) -> Tuple[Path, Path]:
        target = sequence.pop(len(sequence) // 2 + 1)
        if self.max_exposure_ratio < 1.0:
            sequence = self.sample_under_exposure(sequence)

        idx = random.randint(0, len(sequence) - 1)
        return sequence[idx], target

    def darkest_halfexp(self, sequence: List[Path]) -> Tuple[Path, Path]:
        target = sequence.pop(len(sequence) // 2 + 1)
        return sequence[0], target

    def __call__(self, sequence: List[Path], target: Path) -> Tuple[Path, Path]:
        if self.method == PairSelectionMethod.RANDOM_NEXT:
            return self.random_next(sequence)
        elif self.method == PairSelectionMethod.RANDOM_TARGET:
            return self.random_target(sequence, target)
        elif self.method == PairSelectionMethod.HALFEXP_TARGET:
            return self.halfexp_target(sequence, target)
        elif self.method == PairSelectionMethod.RANDOM_HALFEXP:
            return self.random_halfexp(sequence)
        elif self.method == PairSelectionMethod.DARKEST_HALFEXP:
            return self.darkest_halfexp(sequence)
        else:
            raise ValueError(f"unknown selection method: {self.method}")


class SICE(Dataset):
    def __init__(
        self,
        root: Path,
        indices: Optional[List[int]] = None,
        train: bool = True,
        pair_selection_method: PairSelectionMethod = PairSelectionMethod.RANDOM_TARGET,
        max_exposure_ratio: float = 1.0,
        pair_transform: Optional[Callable] = None,
    ):
        self.root = root / ("Train" if train else "Test")
        self.pair_selector = PairSelector(pair_selection_method, max_exposure_ratio)
        self.pair_transform = (
            pair_transform
            if pair_transform is not None
            else A.Compose(
                [ToTensorV2()],
                additional_targets={"target": "image"},
            )
        )

        self.images = sorted(
            (self.root / "Images").glob("*"), key=lambda x: int(x.stem)
        )
        self.targets = sorted(
            (self.root / "Targets").glob("*.[JPG PNG]*"),
            key=lambda x: int(x.stem),
        )

        if indices is not None:
            self.images = [self.images[index] for index in indices]
            self.targets = [self.targets[index] for index in indices]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index: int) -> PairedImageInput:
        image_sequence = sorted(
            self.images[index].glob("*.[JPG PNG]*"), key=lambda x: int(x.stem)
        )

        image_path, target_path = self.pair_selector(
            image_sequence, self.targets[index]
        )

        image = read_image_cv2(image_path)
        target = read_image_cv2(target_path)

        transformed = self.pair_transform(image=image, target=target)
        image, target = transformed["image"], transformed["target"]

        return PairedImageInput(image=image, target=target)


class SICEDataModule(pl.LightningDataModule):
    def __init__(self, config: SICEDatasetConfig):
        super().__init__()
        self.root = Path(config.path)
        self.config = config

        self.train_transform, self.test_transform = load_transforms(config.transform)

    def setup(self, stage: Optional[str] = None):
        n_train_images = len(list((self.root / "Train" / "Images").glob("*")))
        train_indices, val_indices = random_split(
            range(n_train_images), [1 - self.config.val_size, self.config.val_size]
        )

        self.train_ds = SICE(
            self.root,
            indices=train_indices,
            train=True,
            pair_transform=self.train_transform,
            max_exposure_ratio=self.config.max_exposure_ratio,
            pair_selection_method=self.config.train_pair_selection_method,
        )

        self.val_ds = SICE(
            self.root,
            indices=val_indices,
            train=True,
            pair_transform=self.test_transform,
            max_exposure_ratio=self.config.max_exposure_ratio,
            pair_selection_method=self.config.test_pair_selection_method,
        )

        self.test_ds = SICE(
            self.root,
            train=False,
            pair_transform=self.test_transform,
            max_exposure_ratio=self.config.max_exposure_ratio,
            pair_selection_method=self.config.test_pair_selection_method,
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
