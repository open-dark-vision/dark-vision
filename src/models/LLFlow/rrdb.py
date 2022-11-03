"""
Note that this model is based on https://github.com/wyf0912/LLFlow repository.
We slightly changed the code to make it work with our framework.
"""
from typing import Tuple, Union

import torch


class ResidualDenseBlock(torch.nn.Module):
    def __init__(
        self,
        channels: int = 64,
        hidden_channels: int = 32,
        p: float = 0.2,
        kernel: Union[Tuple[int, int], int] = (3, 3),
        bias: bool = True,
        conv_layers: int = 5,
    ):
        super(ResidualDenseBlock, self).__init__()
        self.p = p

        self.inner_convolutions = torch.nn.ModuleList(
            [
                torch.nn.Conv2d(
                    channels + i * hidden_channels,
                    hidden_channels,
                    kernel,
                    padding=1,
                    bias=bias,
                )
                for i in range(conv_layers - 1)
            ]
        )

        self.final_conv = torch.nn.Conv2d(
            channels + (conv_layers - 1) * hidden_channels,
            channels,
            kernel,
            padding=1,
            bias=bias,
        )
        self.activation = torch.nn.LeakyReLU(negative_slope=0.2)

        for net in self.inner_convolutions:
            self.init_weights(net)
        self.init_weights(self.final_conv)

    @staticmethod
    def init_weights(conv_layer: torch.nn.Module, sigma: float = 0.1):
        for module in conv_layer.modules():
            torch.nn.init.kaiming_normal_(module.weight, a=0, mode="fan_in")
            module.weight.data *= sigma
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_out = x
        for conv in self.inner_convolutions:
            x_out = torch.cat((x_out, self.activation(conv(x_out))), 1)
        return x + self.p * self.final_conv(x_out)


class RRDB(torch.nn.Module):
    """Residual in Residual Dense Block"""

    def __init__(self, channels: int, hidden_channels: int = 32, p: float = 0.2):
        super(RRDB, self).__init__()
        self.p = p

        self.sub_modules = torch.nn.Sequential(
            ResidualDenseBlock(channels, hidden_channels),
            ResidualDenseBlock(channels, hidden_channels),
            ResidualDenseBlock(channels, hidden_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.p * self.sub_modules(x)


if __name__ == "__main__":
    from timeit import default_timer as timer

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {DEVICE}")

    model = RRDB(channels=64, hidden_channels=32).to(DEVICE)
    X = torch.zeros((16, 64, 400, 600), device=DEVICE)

    start = timer()
    print(f"Output size: {model(X).size()}")
    stop = timer()

    print(f"RRDB forward time: {stop - start:.2f} sec")
