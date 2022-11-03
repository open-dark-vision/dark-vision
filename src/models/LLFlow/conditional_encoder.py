from typing import Dict, Tuple

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
        """
        TODO:
        call a function which does: HE, color map, noise map and concatenate them all
        """

        feature_maps_1 = self.first_submodule(x)

        block_indexes = [0, 2, 4, 6]
        block_results = {}

        current_feature_map = feature_maps_1
        for index, m in enumerate(self.RRDB_submodule.children()):
            current_feature_map = m(current_feature_map)
            if index in block_indexes:
                block_results[f"feature_maps_{2 + index//2}"] = current_feature_map

        feature_maps_6 = feature_maps_1 + self.trunk_conv(current_feature_map)

        feature_maps_7 = self.first_downsampling(feature_maps_6)

        feature_maps_8 = self.second_downsampling(feature_maps_7)

        color_map = self.fine_tune_color_map(feature_maps_6)

        return {
            "feature_maps_1": feature_maps_1,
            **block_results,
            "feature_maps_6": feature_maps_6,
            "feature_maps_7": feature_maps_7,
            "feature_maps_8": feature_maps_8,
            "color_map": color_map,
        }


if __name__ == "__main__":
    from timeit import default_timer as timer

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {DEVICE}")

    model = ConditionalEncoder(
        channels_in=3, channels_middle=64, rrdb_number=24, rrdb_channels=32
    ).to(DEVICE)
    X = torch.zeros((1, 3, 400, 600), device=DEVICE)

    start = timer()
    output = model(X)
    stop = timer()

    print("Input shape:", X.size())
    for k, v in output.items():
        print(k, v.size())

    print(f"Conditional encoder forward time: {stop - start:.2f} sec")
