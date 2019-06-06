# sub-parts of the U-Net model

import torch
import torch.nn as nn
import torch.nn.functional as F
from MobileNetDenseASPP import InvertedResidual, InvertedV1Residual
class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch, dilation_rate=[1, 1]):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential()
        for i, rate in enumerate(dilation_rate):
            if i!=0:
                in_ch = out_ch
            self.conv.add_module('down_conv_%s'%i, InvertedV1Residual(in_ch, out_ch, stride=1, expand_ratio=2, dilation=rate))
            # nn.Conv2d(in_ch, out_ch, 3, padding=1),
            # nn.BatchNorm2d(out_ch),
            # nn.ReLU(inplace=True),
            # InvertedV1Residual(in_ch, out_ch, stride=1, expand_ratio=2, dilation=1),
            # nn.Conv2d(out_ch, out_ch, 3, padding=1),
            # nn.BatchNorm2d(out_ch),
            # nn.ReLU(inplace=True)
            # InvertedV1Residual(out_ch, out_ch, stride=1, expand_ratio=2, dilation=1),

    def forward(self, x):
        x = self.conv(x)
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch, dilation_rate=[1, 1]):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch, dilation_rate)

    def forward(self, x):
        xout = x
        x = self.conv(x)
        # x = torch.cat([x, xout], dim=1)
        return x, xout


class down(nn.Module):
    def __init__(self, in_ch, out_ch, dilation_rate=[1, 1]):
        super(down, self).__init__()
        # self.mpconv = nn.ModuleList([
        #     nn.MaxPool2d(2),
        #     double_conv(in_ch, out_ch-in_ch, dilation_rate)]
        # )
        self.mpconv = double_conv(in_ch, out_ch-in_ch, dilation_rate)

    def forward(self, x):
        # for idx, m in enumerate(self.mpconv):
        #     if idx == 0:
        #         xout = m(x)
        #         x = xout
        #     else:
        #         x = m(x)
        x_bili = nn.functional.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False)
        x = self.mpconv(x_bili)
        x = torch.cat([x, x_bili], dim=1)
        return x, x_bili


class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=False):
        super(up, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch, in_ch//2, 2, stride=2)

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX//2,
                        diffY // 2, diffY - diffY//2))
        
        # for padding issues, see 
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x
