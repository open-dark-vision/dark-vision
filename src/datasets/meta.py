from typing import NewType, TypedDict

import torch

Image = NewType("Image", torch.Tensor)


class PairedImageInput(TypedDict):
    image: Image
    target: Image


class SixFoldImageInput(TypedDict):
    image: Image
    target: Image
    reflect_high: Image
    reflect_low: Image
    illum_high: Image
    illum_low: Image


class AnnotatedBBoxImageInput(TypedDict):
    """Annotated Bounding Box Image Input.

    Args:
        image: Image tensor.
        labels: shape: (num of objects,).
        bboxes: shape(num of objects, 4),
            bbox features:
            num pixels from left edge, num pixels from top edge, width, height.

    """

    image: Image
    labels: torch.Tensor
    bboxes: torch.Tensor


UnpairedImageInput = PairedImageInput
