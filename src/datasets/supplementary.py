from enum import Enum
from pathlib import Path

from albumentations.pytorch.transforms import ToTensorV2
from torch.utils.data import Dataset

from src.datasets.meta import Image  # noqa: I900
from src.utils.image import read_image_cv2  # noqa: I900


class SupplementaryNames(str, Enum):
    DICM = "dicm"
    FUSION = "fusion"
    LIME = "lime"
    LOW = "low"
    MEF = "mef"
    NPE = "npe"
    VV = "vv"


class Supplementary(Dataset):
    def __init__(self, root: Path, dataset: SupplementaryNames, transform=None):
        self.root = self.select_dataset(root, dataset)
        self.images_paths = list(
            self.root.glob("**/*.[JPG PNG jpg png JPEG jpeg bmp]*")
        )

        self.transform = transform if transform is not None else ToTensorV2()

    def select_dataset(self, root: Path, dataset_name: str) -> Path:
        for dataset in SupplementaryNames:
            if dataset == dataset_name:
                return root / dataset.name

    def __len__(self) -> int:
        return len(self.images_paths)

    def __getitem__(self, index: int) -> Image:
        image_path = self.images_paths[index]
        image = read_image_cv2(image_path)
        image = self.transform(image=image)["image"]

        return {"image": image}
