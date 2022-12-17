import random
from typing import Tuple

import numpy as np
import torch
from torch import nn

from src.models.LLFlow.flow_layers import SqueezeLayer  # noqa: I900


class Interpolate(nn.Module):
    def __init__(
        self,
        scale_factor,
        mode="nearest",
        recompute_scale_factor=True,
    ):
        super(Interpolate, self).__init__()
        self.func = nn.functional.interpolate
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


class Conv2dZeros(nn.Conv2d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=(3, 3),
        stride=1,
        padding=0,
        logscale_factor=3,
    ):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding)

        self.logscale_factor = logscale_factor
        self.register_parameter("logs", nn.Parameter(torch.zeros(out_channels, 1, 1)))

        self.weight.data.zero_()
        self.bias.data.zero_()

    def forward(self, x):
        output = super().forward(x)
        return output * torch.exp(self.logs * self.logscale_factor)


class LLFlowNLL:
    Log2PI = float(np.log(2 * np.pi))

    def __init__(self, p=0.2, random_state=42):
        random.seed(random_state)
        self.p = p
        self.squeeze = SqueezeLayer(factor=8)

    def __call__(self, z, logdet, encoded_color_map, gt):
        reference = (
            encoded_color_map if random.random() > self.p else LLFlowNLL.color_map(gt)
        )
        reference, _ = self.squeeze(reference)

        objective = logdet + LLFlowNLL.log_prob(
            z, mean=reference, log_var=torch.tensor(0.0).to(z.device)
        )
        return -objective / float(np.log(2.0) * gt.size(2) * gt.size(3))

    @staticmethod
    def color_map(image: torch.tensor) -> torch.tensor:
        return image / (torch.sum(image, dim=1, keepdim=True) + 1e-4)

    @staticmethod
    def log_prob(x, mean, log_var):
        return torch.sum(LLFlowNLL.likelihood(x, mean, log_var), dim=(1, 2, 3))

    @staticmethod
    def likelihood(x, mean, log_var):
        return -0.5 * (
            log_var * 2.0
            + ((x - mean) ** 2) / torch.exp(log_var * 2.0)
            + LLFlowNLL.Log2PI
        )


def split_channels(z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    return z[:, ::2, ...], z[:, 1::2, ...]
