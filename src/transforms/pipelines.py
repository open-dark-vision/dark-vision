import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2


def pair_transform_example():
    common_transform = A.Compose(
        [
            A.Rotate(),
            A.RandomCrop(300, 300),
            ToTensorV2(),
        ],
        additional_targets={"target": "image"},
    )

    return common_transform, common_transform


def single_image_transform_example():
    common_transform = A.Compose(
        [
            A.Resize(512, 512),
            ToTensorV2(),
        ]
    )

    return common_transform, common_transform


def supplementary_dataset_transform_example():
    return None, A.Compose(
        [
            A.Resize(512, 512),
            ToTensorV2(),
        ],
        additional_targets={"target": "image"},
    )
