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


def gradient(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    def sub_gradient(_x):
        left_shift_x, right_shift_x = torch.zeros_like(_x), torch.zeros_like(_x)
        left_shift_x[:, :, 0:-1] = _x[:, :, 1:]
        right_shift_x[:, :, 1:] = _x[:, :, 0:-1]
        return 0.5 * (left_shift_x - right_shift_x)
    return sub_gradient(x), sub_gradient(x.transpose(2, 3)).transpose(2, 3)


def histogram_equalization(x: torch.Tensor) -> torch.Tensor:
    return torchvision.transforms.functional.equalize(x)


def color_map(x: torch.Tensor) -> torch.Tensor:
    return x / (x.sum(dim=1, keepdim=True) + 1e-4)


def noise_map(x: torch.Tensor) -> torch.Tensor:
    c_map = color_map(x.exp())
    dx, dy = gradient(c_map)
    return torch.max(dx.abs(), dy.abs())


def image_preprocessing(x: torch.Tensor) -> torch.Tensor:
    hist_eq = histogram_equalization(x).double()
    x = x.double()
    c_map = color_map(x)
    n_map = noise_map(x)
    return torch.cat([x, hist_eq, c_map, n_map])


if __name__ == "__main__":
    X = torch.randint(0, 255, (2, 3, 16, 16), dtype=torch.uint8)
    print(histogram_equalization(X))
