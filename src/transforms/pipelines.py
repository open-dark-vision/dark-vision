import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

from src.configs.base import Transform, TransformConfig  # noqa: I900
from src.transforms.custom_transforms import LLFlowTransform  # noqa: I900


def load_transforms(transform_config: TransformConfig):
    if transform_config.name == Transform.DEVELOPMENT:
        transforms = development_transform(
            image_size=transform_config.image_size,
            pair_transform=transform_config.pair_transform,
        )
    elif transform_config.name == Transform.FLIP:
        transforms = flip_transform(
            image_size=transform_config.image_size,
            pair_transform=transform_config.pair_transform,
        )
    elif transform_config.name == Transform.LLFLOW:
        transforms = (
            LLFlowTransform(train=True, crop_size=transform_config.image_size),
            LLFlowTransform(train=False, crop_size=transform_config.image_size),
        )
    elif transform_config.name == Transform.FLIP_NO_RESIZE:
        transforms = flip_no_resize_transform(
            pair_transform=transform_config.pair_transform,
        )
    else:
        raise ValueError(f"Transform {transform_config.name} not found.")

    return transforms


def development_transform(image_size: int, pair_transform: bool = False):
    transform = A.Compose(
        [
            A.Rotate(p=0.5),
            A.RandomCrop(image_size, image_size),
            ToTensorV2(),
        ],
        additional_targets={"target": "image"} if pair_transform else {},
    )

    return transform, transform


def flip_transform(image_size: int = 256, pair_transform: bool = True):
    train_transform = A.Compose(
        [
            A.RandomCrop(image_size, image_size, always_apply=True),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Normalize(mean=0, std=1),
            ToTensorV2(),
        ],
        additional_targets={"target": "image"} if pair_transform else {},
    )

    test_transform = A.Compose(
        [
            # A.Resize(image_size, image_size, always_apply=True),
            A.Normalize(mean=0, std=1),
            ToTensorV2(),
        ],
        additional_targets={"target": "image"} if pair_transform else {},
    )

    return train_transform, test_transform


def flip_no_resize_transform(pair_transform: bool = True):
    train_transform = A.Compose(
        [
            # A.RandomCrop(image_size, image_size, always_apply=True),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Normalize(mean=0, std=1),
            ToTensorV2(),
        ],
        additional_targets={"target": "image"} if pair_transform else {},
    )

    test_transform = A.Compose(
        [
            # A.Resize(image_size, image_size, always_apply=True),
            A.Normalize(mean=0, std=1),
            ToTensorV2(),
        ],
        additional_targets={"target": "image"} if pair_transform else {},
    )

    return train_transform, test_transform
