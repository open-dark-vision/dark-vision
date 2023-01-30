
import torch
from torch import nn


class Restoration_net(nn.Module):
    def __init__(self):
        super(Restoration_net, self).__init__()

        self.part_one = nn.Sequential(
            nn.Conv2d(4, 32, 3, dilation=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(32, 32, 3, dilation=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

        self.part_two = nn.Sequential(
            nn.MaxPool2d(2, padding='same'),
            nn.Conv2d(32, 64, 3, dilation=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(64, 64, 3, dilation=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

        self.part_three = nn.Sequential(
            nn.MaxPool2d(2, padding='same'),
            nn.Conv2d(64, 128, 3, dilation=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(128, 128, 3, dilation=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

        self.part_four = nn.Sequential(
            nn.MaxPool2d(2, padding='same'),
            nn.Conv2d(128, 256, 3, dilation=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(256, 256, 3, dilation=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

        self.part_five = nn.Sequential(
            nn.MaxPool2d(2, padding='same'),
            nn.Conv2d(256, 512, 3, dilation=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(512, 512, 3, dilation=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

        self.deconv_1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)

        self.part_six = nn.Sequential(
            nn.Conv2d(512, 256, 3, dilation=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(256, 256, 3, dilation=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

        self.deconv_2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)

        self.part_seven = nn.Sequential(
            nn.Conv2d(256, 128, 3, dilation=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(128, 128, 3, dilation=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

        self.deconv_3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)

        self.part_eight = nn.Sequential(
            nn.Conv2d(128, 64, 3, dilation=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(64, 64, 3, dilation=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

        self.deconv_4 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)

        self.part_nine = nn.Sequential(
            nn.Conv2d(64, 32, 3, dilation=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(32, 32, 3, dilation=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

        self.part_ten = nn.Sequential(
            nn.Conv2d(32, 3, 3, dilation=1),
            nn.Sigmoid(),
        )

    def forward(self, Reflect, Illumin):
        x_cat = torch.cat((Reflect, Illumin), 1)
        part1 = self.part_one(x_cat)
        part2 = self.part_two(part1)
        part3 = self.part_three(part2)
        part4 = self.part_four(part3)
        part5 = self.part_five(part4)
        part6 = self.part_six(torch.cat((part4, self.deconv_1(part5)), 1))
        part7 = self.part_seven(torch.cat((part3, self.deconv_2(part6)), 1))
        part8 = self.part_eight(torch.cat((part2, self.deconv_3(part7)), 1))
        part9 = self.part_nine(torch.cat((part1, self.deconv_4(part8)), 1))
        return self.part_ten(part9)

