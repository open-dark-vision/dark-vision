from typing import Tuple

import torch
import torchvision


class Interpolate(torch.nn.Module):
    def __init__(
        self,
        scale_factor,
        mode="nearest",
        recompute_scale_factor=True,
    ):
        super(Interpolate, self).__init__()
        self.func = torch.nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode
        self.recompute_scale_factor = recompute_scale_factor

    def forward(self, x):
        return self.func(
            x,
            scale_factor=self.scale_factor,
            mode=self.mode,
            recompute_scale_factor=self.recompute_scale_factor,
        )
