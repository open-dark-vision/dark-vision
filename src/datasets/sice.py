import random
from pathlib import Path
from typing import Callable, Optional

import albumentations as A
import numpy as np
from albumentations.pytorch.transforms import ToTensorV2
from torch.utils.data import Dataset

from src.configs.base import PairSelectionMethod  # noqa: I900
from src.datasets.meta import PairedImageInput  # noqa: I900
from src.utils.image import read_image_cv2  # noqa: I900


class PairSelector:
    def __init__(self, method: PairSelectionMethod, max_exposure_ratio: float = 1.0):
        self.method = method
        self.max_exposure_ratio = max_exposure_ratio

    def sample_under_exposure(self, sequence: list[Path]) -> tuple[Path, Path]:
        n_images = len(sequence)
        percentages = (np.arange(n_images) + 1) / n_images

        n_images_under_exposure = (percentages <= self.max_exposure_ratio).sum()
        return sequence[:n_images_under_exposure]

    def random_next(self, sequence: list[Path]) -> tuple[Path, Path]:
        if self.max_exposure_ratio < 1.0:
            sequence = self.sample_under_exposure(sequence)

        idx = random.randint(0, len(sequence) - 2)
        return sequence[idx], sequence[idx + 1]

    def random_target(self, sequence: list[Path], target: Path) -> tuple[Path, Path]:
        if self.max_exposure_ratio < 1.0:
            sequence = self.sample_under_exposure(sequence)

        idx = random.randint(0, len(sequence) - 1)
        return sequence[idx], target

    def halfexp_target(self, sequence: list[Path], target: Path) -> tuple[Path, Path]:
        idx = len(sequence) // 2
        return sequence[idx], target

    def __call__(self, sequence: list[Path], target: Path) -> tuple[Path, Path]:
        if self.method == PairSelectionMethod.RANDOM_NEXT:
            return self.random_next(sequence)
        elif self.method == PairSelectionMethod.RANDOM_TARGET:
            return self.random_target(sequence, target)
        elif self.method == PairSelectionMethod.HALFEXP_TARGET:
            return self.halfexp_target(sequence, target)
        else:
            raise ValueError(f"unknown selection method: {self.method}")


class SICE(Dataset):
    def __init__(
        self,
        root: Path,
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
