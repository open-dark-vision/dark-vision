from typing import NewType, TypedDict

import torch

Image = NewType("Image", torch.Tensor)


class PairedImageInput(TypedDict):
    image: Image
    target: Image


UnpairedImageInput = PairedImageInput
