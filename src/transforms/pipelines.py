import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

from src.configs.base import Transform, TransformConfig  # noqa: I900


def load_transforms(transform_config: TransformConfig):
    if transform_config.name == Transform.DEVELOPMENT:
        transforms = development_transform(
            image_size=transform_config.image_size,
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
