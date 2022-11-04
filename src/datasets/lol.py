from pathlib import Path
from typing import Callable, Optional

from torch.utils.data import Dataset
from torchvision.io import read_image

from src.datasets.meta import PairedImageInput  # noqa: I900


class LOL(Dataset):
    def __init__(
        self,
        root: Path,
        train: bool = True,
        pair_transform: Optional[Callable] = None,
        preload: bool = False,
    ):
        self.pair_transform = pair_transform

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
            self.loaded_images_.append(read_image(str(self.image_names[index])))
            self.loaded_targets_.append(read_image(str(self.image_names[index])))

    def __len__(self) -> int:
        return len(self.image_names)

    def __getitem__(self, index: int) -> PairedImageInput:
        image = (
            self.loaded_images_[index]
            if self.loaded
            else read_image(str(self.image_names[index]))
        )
        target = (
            self.loaded_targets_[index]
            if self.loaded
            else read_image(str(self.target_names[index]))
        )

        if self.pair_transform:
            image, target = self.pair_transform(image, target)

        return PairedImageInput(image=image, target=target)
