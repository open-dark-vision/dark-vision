import math

from timm.models.layers import trunc_normal_
from torch import nn

from src.models.IAT.blocks import CBlock_ln, SwinTransformerBlock  # noqa: I900


class LocalNet(nn.Module):
    def __init__(self, in_dim=3, dim=16, number=4, layers_type="ccc"):
        super().__init__()

        self.conv1 = nn.Conv2d(in_dim, dim, 3, padding=1, groups=1)
        self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        block_c = CBlock_ln(dim)
        block_t = SwinTransformerBlock(dim)  # head number

        if layers_type == "ccc":
            blocks1 = [
                CBlock_ln(16, drop_path=0.01),
                CBlock_ln(16, drop_path=0.05),
                CBlock_ln(16, drop_path=0.1),
            ]
            blocks2 = [
                CBlock_ln(16, drop_path=0.01),
                CBlock_ln(16, drop_path=0.05),
                CBlock_ln(16, drop_path=0.1),
            ]
        elif layers_type == "ttt":
            blocks1, blocks2 = [block_t for _ in range(number)], [
                block_t for _ in range(number)
            ]
        elif layers_type == "cct":
            blocks1, blocks2 = [block_c, block_c, block_t], [block_c, block_c, block_t]

        self.mul_blocks = nn.Sequential(*blocks1)
        self.add_blocks = nn.Sequential(*blocks2)

        self.mul_end = nn.Sequential(nn.Conv2d(dim, 3, 3, 1, 1), nn.ReLU())
        self.add_end = nn.Sequential(nn.Conv2d(dim, 3, 3, 1, 1), nn.Tanh())
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, img):
        img1 = self.relu(self.conv1(img))
        # short cut connection
        mul = self.mul_blocks(img1) + img1
        add = self.add_blocks(img1) + img1
        mul = self.mul_end(mul)
        add = self.add_end(add)

        return mul, add
