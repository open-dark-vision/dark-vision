from typing import Callable, Optional, Tuple

import src.transforms.pipelines as pipelines  # noqa: I900

SUPPORTED_TRANSFORMS = {
    "pair_transform_example": pipelines.pair_transform_example,
    "single_image_transform_example": pipelines.single_image_transform_example,
    "supplementary_dataset_transform_example": pipelines.supplementary_dataset_transform_example,  # noqa: E501
}


def load_transforms(
    pipeline_name: str,
) -> Tuple[Optional[Callable], Optional[Callable]]:
    return SUPPORTED_TRANSFORMS[pipeline_name]()
