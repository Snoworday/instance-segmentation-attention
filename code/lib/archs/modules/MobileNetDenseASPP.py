import torch
import torch.nn.functional as F

from torch import nn
from torch.nn import BatchNorm2d as bn

class DenseASPP(nn.Module):
    """
    * output_scale can only set as 8 or 16
    """
    n_filters = [24, 64, 160]
    model_cfg = {
        'bn_size': 4,
        'drop_rate': 0.5,
        'growth_rate': 48,
        'num_init_features': 96,
        'block_config': (6, 12, 36, 24),

        'dropout0': 0.1,
        'dropout1': 0.1,
        'd_feature0': 512,
        'd_feature1': 128,

        'pretrained_path': "./pretrained/densenet161.pth"
    }

    def __init__(self, n_class=19, output_stride=8):
        super(DenseASPP, self).__init__()
        dropout0 = self.model_cfg['dropout0']
        dropout1 = self.model_cfg['dropout1']
        d_feature0 = self.model_cfg['d_feature0']
        d_feature1 = self.model_cfg['d_feature1']

        feature_size = int(output_stride / 4)
        self.features = DilatedMobileNetV2(output_stride=output_stride)
        num_features = self.features.get_num_features()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight.data)

            elif isinstance(m, bn):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, _input):
        feature = self.features(_input)
        out = [feature[0], feature[1], feature[2]]
        return feature


def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )


class InvertedV1Residual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio = 2, dilation=1, with_relu=False):
        super(InvertedV1Residual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        self.use_res_connect = self.stride == 1 and inp == oup

        self.conv = nn.Sequential(
            # dw
            nn.Conv2d(inp, inp, kernel_size=3, stride=stride, padding=dilation,
                      dilation=dilation, groups=inp, bias=False),
            nn.BatchNorm2d(inp),
            nn.ReLU6(inplace=True),
            # pw-linear
            nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        )
        if with_relu:
            self.conv.add_module('relul', nn.ReLU6(inplace=True))

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, dilation=1):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        self.use_res_connect = self.stride == 1 and inp == oup

        self.conv = nn.Sequential(
            # pw
            nn.Conv2d(inp, inp * expand_ratio, 1, 1, 0, bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU6(inplace=True),
            # dw
            nn.Conv2d(inp * expand_ratio, inp * expand_ratio, kernel_size=3, stride=stride, padding=dilation,
                      dilation=dilation, groups=inp * expand_ratio, bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU6(inplace=True),
            # pw-linear
            nn.Conv2d(inp * expand_ratio, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class DilatedMobileNetV2(nn.Module):

    def __init__(self, width_mult=1., output_stride=1):
        super(DilatedMobileNetV2, self).__init__()
        self.num_features = 320
        self.last_channel = 256
        self.scale_factor = int(output_stride / 1)
        scale = self.scale_factor
        # setting of inverted residual blocks
        self.interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1, 1],
            [6, 24, 2, 1, 1],
            [6, 32, 3, 2, 1],
            [6, 64, 4, int(scale), int(2 / scale)],
            [6, 96, 3, 2, int(2 / scale)],
            [6, 160, 3, 1, int(2 / scale)],
            [6, 320, 1, 2, int(2 / scale)],
        ]

        input_channel = int(32 * width_mult)
        self.features = [conv_bn(3, input_channel, 1)]
        # building inverted residual blocks
        for t, c, n, s, dilate in self.interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                if i == 0:
                    self.features.append(InvertedResidual(input_channel, output_channel, s, t, dilation=dilate))
                else:
                    self.features.append(InvertedResidual(input_channel, output_channel, 1, t, dilation=dilate))
                input_channel = output_channel
        self.features.append(conv_1x1_bn(input_channel, self.last_channel))
        self.features = nn.Sequential(*self.features)

    def forward(self, x):
        self.outputs = [3, 10, 16]
        out = []
        childlist = list(self.features.children())
        for i, layer in enumerate(childlist):
            x = layer(x)
            if i in self.outputs:
                out.append(x)
        out.append(x)
        return out

    def get_num_features(self):
        return self.num_features


class _DenseAsppBlock(nn.Sequential):
    """ ConvNet block for building DenseASPP. """

    def __init__(self, input_num, num1, num2, dilation_rate, drop_out, bn_start=True):
        super(_DenseAsppBlock, self).__init__()
        if bn_start:
            self.add_module('norm_1', In(input_num, momentum=0.0003)),

        self.add_module('relu_1', nn.ReLU(inplace=True)),
        self.add_module('conv_1', nn.Conv2d(in_channels=input_num, out_channels=num1, kernel_size=1)),

        self.add_module('norm_2', In(num1, momentum=0.0003)),
        self.add_module('relu_2', nn.ReLU(inplace=True)),
        self.add_module('conv_2', nn.Conv2d(in_channels=num1, out_channels=num2, kernel_size=3,
                                            dilation=dilation_rate, padding=dilation_rate)),

        self.drop_rate = drop_out

    def forward(self, _input):
        feature = super(_DenseAsppBlock, self).forward(_input)

        if self.drop_rate > 0:
            feature = F.dropout2d(feature, p=self.drop_rate, training=self.training)

        return feature


if __name__ == "__main__":
    model = DenseASPP(2)
    print(model)
