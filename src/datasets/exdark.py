from pathlib import Path
from typing import Dict, Optional

import pytorch_lightning as pl
import torch
from albumentations.pytorch.transforms import ToTensorV2
from torch.utils.data import DataLoader, Dataset, random_split

from src.datasets.meta import AnnotatedBBoxImageInput  # noqa: I900
from src.transforms import load_transforms  # noqa: I900
from src.utils.image import read_image_cv2  # noqa: I900

EXDARK_LABEL2ID = {
    "Bicycle": 0,
    "Boat": 1,
    "Bottle": 2,
    "Bus": 3,
    "Car": 4,
    "Cat": 5,
    "Chair": 6,
    "Cup": 7,
    "Dog": 8,
    "Motorbike": 9,
    "People": 10,
    "Table": 11,
}


class ExDark(Dataset):
    def __init__(self, root: Path, train: bool = True, transform=None):
        root = root / ("Train" if train else "Test")
        self.annotation_root = root / "Annotations"
        self.images_paths = [
            path
            for path in root.glob("**/*.[JPG PNG jpg png JPEG jpeg]*")
            if "Annotations" not in str(path)
        ]

        self.transform = transform if transform is not None else ToTensorV2()

    def read_annotation(self, img_path) -> tuple[torch.Tensor, torch.Tensor]:
        annotation_path = (
            self.annotation_root / img_path.parent.name / (img_path.name + ".txt")
        )
        with annotation_path.open() as f:
            objects = f.readlines()[1:]

        labels = []
        bboxes = []
        for obj in objects:
            obj = obj.split(" ")
            labels.append(EXDARK_LABEL2ID[obj[0]])
            bboxes.append([int(bbox_info) for bbox_info in obj[1:5]])

        labels = torch.Tensor(labels)
        bboxes = torch.Tensor(bboxes)

        return labels, bboxes

    def __len__(self) -> int:
        return len(self.images_paths)

    def __getitem__(self, index: int) -> AnnotatedBBoxImageInput:
        image_path = self.images_paths[index]
        image = read_image_cv2(str(image_path))
        labels, bboxes = self.read_annotation(image_path)

        image = self.transform(image=image)["image"]

        return AnnotatedBBoxImageInput(image=image, labels=labels, bboxes=bboxes)

    @staticmethod
    def collate_fn(batch):
        images = [item["image"] for item in batch]
        labels = [item["labels"] for item in batch]
        bboxes = [item["bboxes"] for item in batch]

        return {
            "image": torch.stack(images),
            "labels": labels,
            "bboxes": bboxes,
        }


class ExDarkDataModule(pl.LightningDataModule):
    def __init__(self, config: Dict):
        super().__init__()
        self.data_path = Path(config["path"])
        self.batch_size = config["batch_size"]
        self.val_size = config["val_size"]

        self.pin_memory = config["pin_memory"]
        self.num_workers = config["num_workers"]

        self.train_transform, self.test_transform = load_transforms(config["transform"])

    def setup(self, stage: Optional[str] = None):
        exdark_full = ExDark(
            self.data_path,
            train=True,
            transform=self.train_transform,
        )

        self.exdark_train, self.exdark_val = random_split(
            exdark_full, [1 - self.val_size, self.val_size]
        )
        self.exdark_test = ExDark(
            self.data_path,
            train=False,
            transform=self.test_transform,
        )

    def train_dataloader(self):
        return DataLoader(
            self.exdark_train,
            batch_size=self.batch_size,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
            collate_fn=ExDark.collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.exdark_val,
            batch_size=self.batch_size,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
            collate_fn=ExDark.collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.exdark_test,
            batch_size=self.batch_size,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
            collate_fn=ExDark.collate_fn,
        )

    def predict_dataloader(self):
        return self.test_dataloader()
