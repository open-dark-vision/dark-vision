import torch

from torch import nn

class DecomNet(nn.Module):
    def __init__(self):
        super(DecomNet, self).__init__()

        self.part_one = nn.Sequential(
            nn.Conv2d(3, 32, 3, dilation=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

        self.part_two = nn.Sequential(
            nn.MaxPool2d(2, stride=2, padding=0),
            nn.Conv2d(32, 64, 3, dilation=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

        self.part_three = nn.Sequential(
            nn.MaxPool2d(2, stride=2, padding=0),
            nn.Conv2d(64, 128, 3, dilation=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        self.deconv_1 = nn.ConvTranspose2d(128, 64, 2, stride=2)

        self.conv_4 = nn.Conv2d(128, 64, 3, dilation=1, padding=1) 

        self.deconv_2 = nn.ConvTranspose2d(64, 32, 2, stride=2)

        self.conv_5 = nn.Conv2d(64, 32, 3, dilation=1, padding=1) 

        self.conv_6 = nn.Conv2d(32, 3, 1, dilation=1, padding=0)

        self.reflection_sigmoid = nn.Sigmoid()

        # self.part_four = nn.Sequential(
        #     nn.Conv2d(3, 3, 3, dilation=1),
        #     nn.Sigmoid()
        # )

        self.part_five = nn.Sequential(
            nn.Conv2d(32, 32, 3, dilation=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

        self.part_six = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0), 
            nn.Sigmoid()
        )


    def forward(self, x):
        x = x.transpose(1,3)
        x_1 = self.part_one(x)
        x_2 = self.part_two(x_1)
        x = self.part_three(x_2)
        x = self.conv_4(torch.cat((x_2,self.deconv_1(x)),1))
        x_5 = self.conv_5(torch.cat((x_1,self.deconv_2(x)),1))
        ReflectionOut = self.reflection_sigmoid(self.conv_6(x_5))
        x = self.part_five(x_1)
        IlluminationOut = self.part_six(torch.cat((x,x_5),1)) 
        return ReflectionOut, IlluminationOut
