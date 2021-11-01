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


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
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
        return self.conv(x)


class Concat_down(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.concat_down_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
        # self.conv = DoubleConv(out_channels, out_channels)

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        return self.concat_down_conv(x)


class Concat_up(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        in_channels = int(in_channels / 2 * 5)

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2, x3):  #
        x1 = self.up(x1)
        x2 = self.up(x2)

        # input is CHW
        diffY = x3.size()[2] - x1.size()[2]
        diffX = x3.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x2 = F.pad(x2, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        x = torch.cat([x, x3], dim=1)

        return self.conv(x)


class DBUNet(nn.Module):
    def __init__(self, in_ch, out_ch, f):

        if f == 8:
            filters = [8, 16, 32, 64, 128]
        elif f == 16:
            filters = [16, 32, 64, 128, 256]
        elif f == 32:
            filters = [32, 64, 128, 256, 512]
        else:
            filters = [64, 128, 256, 512, 1024]

        super(DBUNet, self).__init__()
        self.inc = DoubleConv(in_ch, filters[0])
        self.down1 = Down(filters[0], filters[1])
        self.down2 = Down(filters[1], filters[2])
        self.down3 = Down(filters[2], filters[3])
        self.drop3 = nn.Dropout2d(0.5)
        self.down4 = Down(filters[3], filters[4])
        self.drop4 = nn.Dropout2d(0.5)
        self.up4 = Up(filters[4], filters[3], False)
        self.up3 = Up(filters[3], filters[2], False)
        self.up2 = Up(filters[2], filters[1], False)
        self.up1 = Up(filters[1], filters[0], False)
        self.outc = OutConv(filters[0], out_ch)

        # border start
        self.b_inc = DoubleConv(in_ch, filters[0])
        self.b_down1 = Down(filters[0], filters[1])
        self.b_down2 = Down(filters[1], filters[2])
        self.b_down3 = Down(filters[2], filters[3])
        self.b_up3 = Up(filters[3], filters[2], False)
        self.b_up2 = Up(filters[2], filters[1], False)
        self.b_up1 = Up(filters[1], filters[0], False)
        self.b_outc = OutConv(filters[0], out_ch)
        # border end

        # concat
        self.concat_down1 = Concat_down(filters[0] * 2, filters[1])
        self.concat_down2 = Concat_down(filters[1] * 2, filters[2])
        self.concat_down3 = Concat_down(filters[2] * 2, filters[3])

        self.concat_up3 = Concat_up(filters[3], filters[2])
        self.concat_up2 = Concat_up(filters[2], filters[1])
        self.concat_up1 = Concat_up(filters[1], filters[0])

        self.outc = OutConv(filters[0], out_ch)
        # concat

    def forward(self, inputs):
        # border
        b_x1 = self.b_inc(inputs)
        b_x2 = self.b_down1(b_x1)
        b_x3 = self.b_down2(b_x2)
        b_x4 = self.b_down3(b_x3)
        b_up3_x = self.b_up3(b_x4, b_x3)
        b_up2_x = self.b_up2(b_up3_x, b_x2)
        b_up1_x = self.b_up1(b_up2_x, b_x1)
        logits_Border = self.b_outc(b_up1_x)
        # print(x1.size())

        x1 = self.inc(inputs)
        # x2 = self.down1(x1)
        x2 = self.concat_down1(x1, b_x1)
        # x3 = self.down2(x2)
        x3 = self.concat_down2(x2, b_x2)

        # x4 = self.down3(x3)
        x4 = self.concat_down3(x3, b_x3)

        x5 = self.down4(x4)
        up4_x = self.up4(x5, x4)
        up3_x = self.concat_up3(up4_x, b_x4, x3)
        up2_x = self.concat_up2(up3_x, b_up3_x, x2)
        up1_x = self.concat_up1(up2_x, b_up2_x, x1)

        logits_Unet = self.outc(up1_x)
        # print(logits_Unet.size())
        return torch.sigmoid(logits_Unet), torch.sigmoid(logits_Border)
