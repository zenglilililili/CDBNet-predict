""" Full assembly of the parts to form the complete network """
import torch
import torch.nn as nn
import torch.nn.functional as F

from tensorboardX import SummaryWriter


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


# class Up(nn.Module):
#     """Upscaling then double conv"""
#
#     def __init__(self, in_channels, out_channels, bilinear=False, picel_shuffle=2):
#         super().__init__()
#
#         # if bilinear, use the normal convolutions to reduce the number of channels
#         if bilinear:
#             self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
#             self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
#             self.pixel_shuffle = nn.PixelShuffle(picel_shuffle)
#         else:
#             self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
#             self.conv = DoubleConv(in_channels, out_channels)
#             self.pixel_shuffle = nn.PixelShuffle(picel_shuffle)
#
#     def forward(self, x1, x2):
#         x1 = self.pixel_shuffle(x1)
#         x1 = torch.cat([x1, x1], dim=1)
#
#         x = torch.cat([x2, x1], dim=1)
#         return self.conv(x)


#
class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=False):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return torch.sigmoid(self.conv(x))


class ConcatConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.Triple_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, in_channels // 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels // 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 8, out_channels, kernel_size=1, padding=0)
        )
        # self.conv = DoubleConv(out_channels, out_channels)

    def forward(self, x1, x2):
        # x1 = self.conv(x1)
        # x2 = self.conv(x2)
        x = torch.cat([x1, x2], dim=1)
        return torch.sigmoid(self.Triple_conv(x))


class doubleUNet(nn.Module):
    def __init__(self, in_ch, out_ch, f):
        # # filters = [32, 64, 128, 256, 512]
        # filters = [64, 128, 256, 512, 1024]
        # # filters = [16, 32, 64, 128, 256]

        if f == 8:
            filters = [8, 16, 32, 64, 128]
        elif f == 16:
            filters = [16, 32, 64, 128, 256]
        elif f == 32:
            filters = [32, 64, 128, 256, 512]
        else:
            filters = [64, 128, 256, 512, 1024]

        super(doubleUNet, self).__init__()
        self.inc = DoubleConv(in_ch, filters[0])

        self.down1 = Down(filters[0], filters[1])

        self.down2 = Down(filters[1], filters[2])
        self.x2_up = Up(filters[1], filters[0])
        self.down3 = Down(filters[2], filters[3])

        self.x3_up = Up(filters[2], filters[1])
        self.drop3 = nn.Dropout2d(0.5)

        self.x4_up = Up(filters[3], filters[2])

        self.down4 = Down(filters[3], filters[4])
        self.drop4 = nn.Dropout2d(0.5)
        self.up1 = Up(filters[4], filters[3], False)

        self.up2 = Up(filters[3], filters[2], False)
        self.up3 = Up(filters[2], filters[1], False)
        self.up4 = Up(filters[1], filters[0], False)

        # self.outc = OutConv(filters[0], out_ch)
        self.out_border = OutConv(filters[0], out_ch)
        self.concatConv = ConcatConv(2 * filters[0], out_ch)

    def forward(self, x):
        x1 = self.inc(x)
        # print(x1.size())

        x2 = self.down1(x1)

        x3 = self.down2(x2)

        x4 = self.down3(x3)

        x4_up = self.x4_up(x4, x3)

        x3_up = self.x3_up(x4_up, x2)

        x2_up = self.x2_up(x3_up, x1)

        # print(x2_up.size())
        out_border = self.out_border(x2_up)

        x4 = self.drop3(x4)

        x5 = self.down4(x4)
        x5 = self.drop4(x5)
        x = self.up1(x5, x4)
        # logits2 = self.out2(x)
        x = self.up2(x, x3)
        # logits3 = self.out3(x)
        x = self.up3(x, x2)
        # logits4 = self.out4(x)
        x = self.up4(x, x1)

        # out_Unet = self.outc(x)
        # print(x2_up.size())
        # print(x.size())
        out_end = self.concatConv(x2_up, x)
        # out_end = self.concatConv(out_Unet, out_border)
        # print(out_border.size())
        # print(out_Unet.size())
        # print(out_end.size())
        return out_end, out_border
