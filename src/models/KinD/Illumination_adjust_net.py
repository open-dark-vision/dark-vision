
import torch
from torch import nn

class Illumination_adjust_net(nn.Module):
    def __init__(self):
        super(Illumination_adjust_net, self).__init__()

        self.part_one = nn.Sequential(
            nn.Conv2d(2, 32, 3, dilation=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )
        self.part_two = nn.Sequential(
            nn.Conv2d(32, 32, 3, dilation=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )
        self.part_3 = nn.Sequential(
            nn.Conv2d(32, 32, 3, dilation=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )
        self.part_4 = nn.Sequential(
            nn.Conv2d(32, 1, 3, dilation=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Sigmoid(),
        )

    def forward(self, x, x_ratio):
        x_cat = torch.cat((x, x_ratio), 1)
        x = self.part_one(x_cat)
        x = self.part_two(x)
        x = self.part_3(x)
        x = self.part_4(x)

        return x