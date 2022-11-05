from pathlib import Path

import torch
from torch.utils.data import Dataset

from src.datasets.meta import AnnotatedBBoxImageInput  # noqa: I900
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
        super().__init__()
        root = root / ("Train" if train else "Test")
        self.annotation_root = root / "Annotations"
        self.images_paths = [
            path
            for path in root.glob("**/*.[JPG PNG jpg png JPEG jpeg]*")
            if "Annotations" not in str(path)
        ]

        self.transform = transform

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

        if self.transform is not None:
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
