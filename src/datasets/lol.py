from pathlib import Path
from typing import Callable, Optional

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from torch.utils.data import Dataset

from src.datasets.meta import PairedImageInput  # noqa: I900
from src.utils.image import read_image_cv2  # noqa: I900


class LOL(Dataset):
    def __init__(
        self,
        root: Path,
        train: bool = True,
        pair_transform: Optional[Callable] = None,
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
