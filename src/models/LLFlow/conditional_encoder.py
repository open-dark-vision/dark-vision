from typing import Any, Dict, Tuple

import torch

from src.models.LLFlow.rrdb import RRDB  # noqa: I900
from src.models.LLFlow.utils import Interpolate  # noqa: I900


class ConditionalEncoder(torch.nn.Module):
    def __init__(
        self,
        channels_in: int = 12,
        channels_middle: int = 64,
        channels_out: int = 3,
        rrdb_number: int = 24,
        rrdb_channels: int = 32,
        kernel: Tuple[int, int] = (3, 3),
        **kwargs: Any,
    ):
        super(ConditionalEncoder, self).__init__()

        self.first_submodule = torch.nn.Sequential(
            torch.nn.Conv2d(channels_in, channels_middle, kernel, padding=1),
            torch.nn.LeakyReLU(negative_slope=0.2),
            torch.nn.Conv2d(channels_middle, channels_middle, kernel, padding=1),
            torch.nn.MaxPool2d(2),
        )

        self.RRDB_submodule = torch.nn.Sequential(
            *[RRDB(channels_middle, rrdb_channels) for _ in range(rrdb_number)]
        )

        self.trunk_conv = torch.nn.Conv2d(
            channels_middle, channels_middle, kernel, padding=1
        )

        self.first_downsampling = torch.nn.Sequential(
            Interpolate(scale_factor=0.5, mode="bilinear"),
            torch.nn.Conv2d(channels_middle, channels_middle, kernel, padding=1),
        )

        self.second_downsampling = torch.nn.Sequential(
            torch.nn.LeakyReLU(negative_slope=0.2),
            Interpolate(scale_factor=0.5, mode="bilinear"),
            torch.nn.Conv2d(channels_middle, channels_middle, kernel, padding=1),
        )

        self.fine_tune_color_map = torch.nn.Sequential(
            Interpolate(scale_factor=2),
            torch.nn.Conv2d(channels_middle, channels_out, kernel_size=(1, 1)),
            torch.nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        feats_pre_rrdb = self.first_submodule(x)

        feats_post_rrdb = self.RRDB_submodule(feats_pre_rrdb)

        feature_maps_1 = feats_pre_rrdb + self.trunk_conv(feats_post_rrdb)

        feature_maps_2 = self.first_downsampling(feature_maps_1)

        feature_maps_3 = self.second_downsampling(feature_maps_2)

        color_map = self.fine_tune_color_map(feature_maps_1)

        return {
            "feature_maps_1": feature_maps_1,
            "feature_maps_2": feature_maps_2,
            "feature_maps_3": feature_maps_3,
            "color_map": color_map,
        }
