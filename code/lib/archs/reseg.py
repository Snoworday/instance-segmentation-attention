import torch
import torch.nn as nn
from modules.vgg16 import SkipVGG16
from modules.unet.unet_model import UNet
# from modules.renet import ReNet
from modules.attenet2 import DecoderLayer
from modules.dcgan_decoder import DcganDecoder
from modules.MobileNetDenseASPP import DenseASPP, _DenseAsppBlock
from modules.utils import make_position_encoding, AttentionLayer
import numpy as np
import config
class ReSeg(nn.Module):

    r"""ReSeg Module (with modifications) as defined in 'ReSeg: A Recurrent
    Neural Network-based Model for Semantic Segmentation'
    (https://arxiv.org/pdf/1511.07053.pdf).

    * VGG16 with skip Connections as base network
    * Two ReNet layers
    * Two transposed convolutional layers for upsampling
    * Three heads for semantic segmentation, instance segmentation and
        instance counting.

    Args:
        n_classes (int): Number of semantic classes
        use_instance_seg (bool, optional): If `False`, does not perform
            instance segmentation. Default: `True`
        pretrained (bool, optional): If `True`, initializes weights of the
            VGG16 using weights trained on ImageNet. Default: `True`
        use_coordinates (bool, optional): If `True`, adds coordinate
            information to input image and hidden state. Default: `False`
        usegpu (bool, optional): If `True`, runs operations on GPU
            Default: `True`

    Shape:
        - Input: `(N, C_{in}, H_{in}, W_{in})`
        - Output:
            - Semantic Seg: `(N, N_{class}, H_{in}, W_{in})`
            - Instance Seg: `(N, 32, H_{in}, W_{in})`
            - Instance Cnt: `(N, 1)`

    Examples:
        >>> reseg = ReSeg(3, True, True, True, False)
        >>> input = torch.randn(8, 3, 64, 64)
        >>> outputs = reseg(input)

        >>> reseg = ReSeg(3, True, True, True, True).cuda()
        >>> input = torch.randn(8, 3, 64, 64).cuda()
        >>> outputs = reseg(input)
    """

    def __init__(self, n_classes, use_instance_seg=True,
                                   pretrained=True,
                 use_coordinates=False, use_wae=True, usegpu=True, training=True):
        super(ReSeg, self).__init__()
        self.backbone = 'Unet'

        self.n_classes = n_classes
        self.use_instance_seg = use_instance_seg

        self.base = UNet(n_channels=21)
        self.use_wae = use_wae
        self.training = training


        self.decoder = DecoderLayer()
        # if self.use_wae:
        #     self.decoder = DcganDecoder(decoder_opt)
        # Decoder
        if self.backbone == 'Unet':
            # Semantic Segmentation
            self.channelAttend = AttentionLayer(self.base.n_filters)
            self.sem_seg_output = nn.Conv2d(self.base.n_filters,
                                            self.n_classes, kernel_size=(1, 1),
                                            stride=(1, 1))
            # Instance Segmentation
            if self.use_instance_seg:
                self.ins_seg_output_1 = nn.Sequential(
                    nn.Conv2d(self.base.n_filters, self.base.n_filters,
                              kernel_size=(3, 3),
                              stride=(1, 1), padding=(1, 1),
                              groups=self.base.n_filters),
                    nn.BatchNorm2d(self.base.n_filters),
                    nn.ReLU6(),
                    nn.Conv2d(self.base.n_filters, config.d_model,  # * 2
                              kernel_size=(1, 1),
                              stride=(1, 1)),
                    nn.BatchNorm2d(config.d_model),
                    nn.ReLU6()
                )
                self.ins_seg_output_2 = nn.Sequential(
                    nn.Conv2d(config.d_model, config.d_model*2, kernel_size=(1, 1), stride=(1, 1)),
                    nn.BatchNorm2d(config.d_model*2),
                    nn.ReLU6(),
                    nn.Conv2d(config.d_model*2, config.d_model*2, kernel_size=(3, 3), stride=(1, 1),
                              padding=(1, 1),
                              groups=config.d_model*2),
                    nn.BatchNorm2d(config.d_model*2),
                    nn.ReLU6(),
                    nn.Conv2d(config.d_model*2, config.d_model, kernel_size=(1, 1), stride=(1, 1)),
                    nn.BatchNorm2d(config.d_model)
                )
                # 加个1*1
                self.positioin_encoding = None

    def forward(self, training, *_input):
        if len(_input) == 4:
            x, sem_seg_target, ins_seg_target, N = _input
        else:
            x = _input[0]
        if self.backbone == 'Unet':
            x_dec, *X = self.base(x)

        # Semantic Segmentation
        x_att = self.channelAttend(x_dec)
        sem_seg_out = self.sem_seg_output(x_att)
        if len(_input) == 4:
            sem_seg_argmax = sem_seg_target.argmax(1).unsqueeze(1).float()
        else:
            sem_seg_argmax = sem_seg_out.argmax(1).unsqueeze(1).float()
        if self.use_instance_seg:
            x_enc = self.ins_seg_output_1(x_dec)
            x_enc = self.ins_seg_output_2(x_enc) + x_enc
            if x_enc.shape[0]==1:
                a = 1
            ins_cost, criterion, ins_ce_loss, ins_dice_loss = self.decoder(x_enc, sem_seg_argmax, ins_seg_target, N, training, X)

            return sem_seg_out, sem_seg_argmax, ins_cost, criterion, ins_ce_loss, ins_dice_loss #, cluster
        else:
            return sem_seg_out, sem_seg_argmax

    def set_position_encoding(self, ins_seg_out):
            b, n_units, h, w = ins_seg_out.shape
            h_vec = np.tile(make_position_encoding(np, 1, h, n_units // 2, f=10000.)[:, :, :, np.newaxis], (1, 1, 1, w))
            w_vec = np.tile(make_position_encoding(np, 1, w, n_units // 2, f=10000.)[:, :, np.newaxis, :], (1, 1, h, 1))
            vec = np.concatenate([h_vec, w_vec], axis=1)
            return torch.from_numpy(vec).cuda()

