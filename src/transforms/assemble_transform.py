from typing import Dict, List, Tuple

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

import src.transforms.custom_transforms as custom  # noqa: I900

SUPPORTED_TRANSFORMS = {
    "affine": A.Affine,
    "gaussian_blur": A.GaussianBlur,
    "gray_scale": A.ToGray,
    "histogram_equalize": A.Equalize,
    "horizontal_flip": A.HorizontalFlip,
    "image_histogram_equalize": custom.ImageEqualize,
    "random_crop": A.RandomCrop,
    "resize": A.Resize,
    "to_tensor": ToTensorV2,
    "rotate": A.Rotate,
    "vertical_flip": A.VerticalFlip,
}


def extract_transform_properties(transform_config: Dict) -> Tuple[str, Dict]:
    name = transform_config.pop("name")
    return name, transform_config


def create_transform(pipeline: List, additional_targets: List) -> A.Compose:
    transforms = []
    for transform_config in pipeline:
        name, params = extract_transform_properties(dict(transform_config))
        transforms.append(SUPPORTED_TRANSFORMS[name](**params))

    additional_targets = additional_targets if additional_targets else []
    return A.Compose(
        transforms,
        additional_targets={add_target: "image" for add_target in additional_targets},
    )
