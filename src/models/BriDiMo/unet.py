""" Based on: https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_model.py """  # noqa: E501
import torch.nn as nn

from .unet_parts import DoubleConv, Down, MixConv, OutConv, Up


class ConditionalUNet(nn.Module):
    def __init__(self, n_channels=9, bilinear=False):
        super(ConditionalUNet, self).__init__()
        self.n_channels = n_channels
        self.bilinear = bilinear

        factor = 2 if bilinear else 1

        self.inc = DoubleConv(n_channels, 64)

        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024 // factor)

        self.mix1 = MixConv(64 + 2, 64)
        self.mix2 = MixConv(128 + 2, 128)
        self.mix3 = MixConv(256 + 2, 256)
        self.mix4 = MixConv(512 + 2, 512)
        self.mix5 = MixConv(1024 + 2, 1024)

        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)

        self.outc = OutConv(64, 3)

    def forward(self, x, source_lightness, target_lightness):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(
            self.mix5(x5, source_lightness, target_lightness),
            self.mix4(x4, source_lightness, target_lightness),
        )
        x = self.up2(x, self.mix3(x3, source_lightness, target_lightness))
        x = self.up3(x, self.mix2(x2, source_lightness, target_lightness))
        x = self.up4(x, self.mix1(x1, source_lightness, target_lightness))

        return self.outc(x)
