from pathlib import Path

from albumentations.pytorch.transforms import ToTensorV2
from torch.utils.data import Dataset

from src.configs.base import SupplementaryDataset  # noqa: I900
from src.datasets.meta import Image  # noqa: I900
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
