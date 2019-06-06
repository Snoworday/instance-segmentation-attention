import torch
from torch import nn


class DcganDecoder(nn.Module):
    def __init__(self, opt):
        super(DcganDecoder, self).__init__()
        self.coding = opt['coding']
        self.num_units = opt['num_units']
        self.num_layers = opt['num_layers']
        self.out_shape = opt['out_shape']
        self.height = self.out_shape[0] // 2**(self.num_layers - 1)
        self.width = self.out_shape[1] // 2**(self.num_layers - 1)
        self.axis_p_batch = self.num_units*self.height*self.width

        self.linear = nn.Linear(self.coding, self.axis_p_batch, bias=True)
        self.relu1 = nn.ReLU()

        self.layer_x = []
        self.bn = []
        self.relu = []
        seq = []
        num_units = self.num_units
        for i in range(self.num_layers - 1):
            seq.append(nn.ConvTranspose2d(num_units, num_units//2, kernel_size=(5, 5), padding=2, output_padding=1,
                                         stride=(2, 2)))
            seq.append(nn.InstanceNorm2d(num_units//2, affine=True))
            seq.append(nn.ReLU())
            num_units = num_units//2
        # _ = [i.cuda() for i in seq]
        self.conv_seq = nn.Sequential(*seq)
        self.last_h = nn.ConvTranspose2d(num_units, self.out_shape[2], kernel_size=(5, 5), padding=2,
                                         stride=(1, 1))
        self.reconstruction = nn.Sigmoid()

    def forward(self, input):
        h0 = self.linear(input)
        batch = h0.size()[0]
        h0 = h0.view(batch, self.num_units, self.height, self.width)
        h0 = self.relu1(h0)
        # layer_x = h0
        # print(layer_x.shape)
        # for m in self.seq:
        #     layer_x = m(layer_x)
        #     print(m)
        #     print(layer_x.shape)
        layer_x = self.conv_seq(h0)
        last_h = self.last_h(layer_x)
        reconstruction = self.reconstruction(last_h)
        reconstruction = reconstruction.squeeze(1)
        return reconstruction

