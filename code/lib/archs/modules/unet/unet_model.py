# full assembly of the sub-parts to form the complete net

import torch.nn.functional as F

from .unet_parts import *
import config
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes=None):
        super(UNet, self).__init__()
        self.inc = inconv(n_channels, 32, [1, 1])
        self.down1 = down(32, 64, [1, 1])
        self.down2 = down(64, 128, [1, 1])
        self.down3 = down(128, 256, [1, 1])
        self.down4 = down(256, 512, [1, 1])
        self.up1 = up(512, 256)
        self.up2 = up(256, 128)
        self.up3 = up(128, 64)
        self.up4 = up(64, 32)
        # self.outc = outconv(64, n_classes)

        self.n_filters = 32

    def forward(self, x):
        x1, x_out1 = self.inc(x)
        x2, x_out2 = self.down1(x1)
        x3, x_out3 = self.down2(x2)
        x4, x_out4 = self.down3(x3)
        x5, x_out5 = self.down4(x4)
        x_4 = self.up1(x5, x4)
        x_3 = self.up2(x_4, x3)
        x_2 = self.up3(x_3, x2)
        x_1 = self.up4(x_2, x1)
        # x = self.outc(x)
        if config.use_encode:
            # return x_1, x_out1, x_out2, x_out3, x_out4, x_out5
            return x_1, x1, x2, x3, x4, x5
        else:
            return x_1, x_1, x_2, x_3, x_4, x5  # F.sigmoid(x)