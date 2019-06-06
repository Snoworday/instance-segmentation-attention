import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import config
from torch.nn import BatchNorm2d as bn
from torch.nn.parameter import Parameter
import math
from MobileNetDenseASPP import InvertedResidual, InvertedV1Residual, conv_1x1_bn
import random
column_position = torch.linspace(0, 255,256).unsqueeze(0).expand(256, -1).cuda()
row_position = torch.linspace(0, 255,256).unsqueeze(1).expand(-1, 256).cuda()
position = [row_position, column_position]

class Encoder(nn.Module):
    def __init__(self, num_layers, asppList, d_model, d_k, d_v, d_inner, n_head):
        super(Encoder, self).__init__()
        self.num_layers = num_layers
        AttenBlocks = []

        for i in range(num_layers):
            # aspp = _AttenAsppBlock(asppList[i], d_model, d_k, d_v, d_inner, n_head) # input_num, num1, num2, dilation_rate, drop_out
            aspp = _DenseAsppBlock(config.d_model + i*config.d_features1, num1=config.d_features0, num2=config.d_features1, dilation_rate=asppList[i],
                                   drop_out=config.dropout0, bn_start=i!=0)
            AttenBlocks.append(aspp)
        self.AttenSq = nn.ModuleList(AttenBlocks)
        self.last = nn.Sequential(
            nn.Dropout2d(p=config.dropout1),
            nn.Conv2d(in_channels=config.d_model+num_layers*config.d_features1, out_channels=config.d_model, kernel_size=1, padding=0)
        )

    def forward_attn(self, input, mask):
        enc_output = input
        for layer in self.AttenSq:
            enc_output = layer(enc_output, mask)
        return enc_output

    def forward(self, input, mask):
        features = input
        for layer in self.AttenSq:
            features = features * mask
            aspp = layer(features)
            features = torch.cat((aspp, features), dim=1)
        features = features * mask
        out = self.last(features)
        return out


class Decoder(nn.Module):
    def __init__(self, num_layers, d_model, d_inner, n_head, d_k, d_v):
        super(Decoder, self).__init__()
        self.num_layers = num_layers

        self.layer_stack = nn.ModuleList([
            DecoderLayer(d_model, d_inner, n_head, d_k, d_v, last= (i==num_layers-1))
            for i in range(num_layers)])
        self.linear = None

    def forward(self, input, enc_output, mask):
        dec_output = input.unsqueeze(1)
        b, c, h, w = enc_output.shape
        enc_output = enc_output.view(b, c, -1)#.permute(0, 2, 1)

        # for layer in self.layer_stack:
        #     dec_output, dec_slf_attn, dec_enc_attn = layer(dec_output, enc_output, mask)  # mask: 0, 1, 0

        input = input.unsqueeze(1)
        dec_output = torch.sigmoid(torch.bmm(input, enc_output)).squeeze(1)
        return dec_output


class _AttenAsppBlock(nn.Module):
    """ ConvNet block for building DenseASPP. """

    def __init__(self, dilation_rate, d_model, d_k, d_v, d_inner, n_head, bn_start=True):
        super(_AttenAsppBlock, self).__init__()
        self.d = dilation_rate
        self.pad = (self.d, self.d, self.d, self.d)
        self.d_v = d_v
        self.n_head = n_head
        self.attention = _ScalePDAttention( d_k, d_v, d_model, dilation_rate)

        # ForwardLayer
        self.w1 = nn.Conv2d(in_channels=d_model, out_channels=d_inner, kernel_size=(1, 1), stride=(1, 1))
        self.act = nn.LeakyReLU(0.01)
        self.w2 = nn.Conv2d(in_channels=d_inner, out_channels=d_model, kernel_size=(1, 1), stride=(1, 1))
        self.layer_norm = nn.InstanceNorm2d(d_model)

        # self.conv_1 = nn.Conv2d(in_channels=input_num*9, out_channels=input_num*9, kernel_size=3, dilation=dilation_rate, padding=dilation_rate, groups=input_num*9, bias=False)
        #
        # self.conv_1.weight = prodTemplate(dilation_rate, input_num)
        # self.conv_1.weight.requires_grad = False
        # self.drop_rate = drop_out

    def forward(self, _input, mask=None):
        '''mask: b, 256, 256'''
        #`feature = super(_AttenAsppBlock, self).forward(_input)
        b, c, h, w = _input.shape
        nomask = 1 - mask  # nomask: 周围1， 中间0
        # input_pad = F.pad(_input, self.pad, 'constant', 0)
        # merge = [input_pad[:, :, i//3*self.d:i//3*self.d+h, i % 3*self.d:i % 3*self.d+w] for i in range(9)]
        # merge = torch.cat(merge, dim=1)
        # input = _input.permute(0, 2, 3, 1)  # b, h, w, c
        # merge = merge.permute(0, 2, 3, 1)   # b, h, w, 9c
        # q = _input.view(b*h*w, c)
        # kv = merge.view(b*h*w, 9*c)

        atten = self.attention(_input, _input, nomask)    # b, d_model, h, w
        #   positionwiseFeedForward
        feedF = self.w2(self.act(self.w1(atten)))
        result = self.layer_norm(feedF + atten)

        # atten = self.ln_1(atten)
        # # atten = atten.view(b*h*w, self.v_dim)
        # #   feed_forward
        # e = self.w1(atten)
        # e = self.act(e)
        # e = self.w2(e)
        # e = e + atten
        # # e = e.view
        # e = self.ln_2(e)    # bhw, out_dim
        # e = e.view(b, h, w, self.v_dim).permute(0, 3, 1, 2)
        # result = torch.cat([e, _input], dim=1)


        # input_ = input.view(*input.shape, 1) # b, h, w, c, 1
        # merge_ = merge.view(*merge.shape[:-1], 9, c)   # b, h, w, 9, c
        # inner = torch.matmul(merge_, input_)
        # exp_inner = torch.exp(inner)    # b, h, w, 9, 1
        # exp_inner_sum = torch.sum(exp_inner, dim=3, keepdim=True) + 1e-8    # b, h, w, 1, 1
        # atten_p = exp_inner / exp_inner_sum
        # atten = merge_ * atten_p
        # result = torch.sum(atten, dim=3)   # b, h, w, c
        # result = result.permute(0, 3, 1, 2)
        return result


class DecoderLayer(nn.Module):
    ''' Compose with three layers '''
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1, last=False):
        super(DecoderLayer, self).__init__()
        self.last = last
        if last:
            n_head = 1

        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.enc_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, dec_input, enc_output, mask):
        mask = mask.unsqueeze(1).byte()
        slf_attn_mask = 1 - mask    # 101
        dec_output, dec_slf_attn = self.slf_attn(
            dec_input, dec_input, dec_input, mask=None)
        # dec_output *= mask

        dec_output, dec_enc_attn = self.enc_attn(
            dec_output, enc_output, enc_output, mask=slf_attn_mask, last=self.last)
        # dec_output *= mask
        if not self.last:
            dec_output = self.pos_ffn(dec_output)
        # dec_output *= mask

        return dec_output, dec_slf_attn, dec_enc_attn


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)

        self.dropout = nn.Dropout(dropout)


    def forward(self, q, k, v, mask=None, last=False):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k) # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k) # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v) # (n*b) x lv x dv
        if mask is not None:
            mask = mask.repeat(n_head, 1, 1) # (n*b) x .. x ..
        if not last:
            output, attn = self.attention(q, k, v, mask=mask)

            output = output.view(n_head, sz_b, len_q, d_v)
            output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1) # b x lq x (n*dv)

            output = self.dropout(self.fc(output))
            output = self.layer_norm(output + residual)
            return output, attn
        else:
            correlation = self.attention(q, k, v, mask=mask, last=last)
            correlation = torch.sigmoid(correlation)
            correlation = correlation.squeeze(1)
            return correlation, None



class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Conv1d(d_in, d_hid, 1) # position-wise
        self.w_2 = nn.Conv1d(d_hid, d_in, 1) # position-wise
        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        output = x.transpose(1, 2)
        output = self.w_2(F.relu(self.w_1(output)))
        output = output.transpose(1, 2)
        output = self.dropout(output)
        output = self.layer_norm(output + residual)
        return output

class _ScalePDAttention(nn.Module):
    def __init__(self, d_k, d_v, d_model, dilation_rate, n_head=2):
        super(_ScalePDAttention, self).__init__()
        self.qk_w = nn.Conv2d(in_channels=d_model//n_head, out_channels=2*d_k, kernel_size=(1, 1), stride=(1, 1))
        self.v_w = nn.Conv2d(in_channels=d_model//n_head, out_channels=d_v, kernel_size=(1, 1), stride=(1, 1))
        self.fc = nn.Conv2d(in_channels=n_head*d_v, out_channels=d_model, kernel_size=(1, 1), stride=(1, 1))
        self.layer_norm = nn.InstanceNorm2d(d_model)

        self.d = dilation_rate
        self.pad = (self.d, self.d, self.d, self.d)
        self.d_k = d_k
        self.n_head = n_head
        self.d_v = d_v
        self.d_model = d_model

        nn.init.normal_(self.qk_w.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.v_w.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))
        nn.init.xavier_normal_(self.fc.weight)

    def forward(self, qk, v, nomask=None):
        '''qk: b, c, h, w'''
        residual = qk
        qk = qk.view(qk.shape[0]*self.n_head, qk.shape[1]//self.n_head, qk.shape[2], qk.shape[3])
        v = v.view(v.shape[0]*self.n_head, v.shape[1]//self.n_head, v.shape[2], v.shape[3])
        nomask = nomask.repeat(self.n_head, 1, 1, 1)
        b, c, h, w = qk.shape   #这里的b是b*n_head，c是c/n_head

        QK = self.qk_w(qk)
        V = self.v_w(v)
        Q, K = torch.split(QK, [self.d_k, self.d_k], dim=1)   #b, c/v, h, w

        #   Q = K = qk
        K_pad = F.pad(K, self.pad, 'constant', 0)
        V_pad = F.pad(V, self.pad, 'constant', 0)
        nomask_pad= F.pad(nomask, self.pad, 'constant', 0)
        K_ = torch.cat([K_pad[:, :, i//3*self.d:i//3*self.d+h, i % 3*self.d:i % 3*self.d+w] for i in range(9)], dim=1)  # b, 9c, h, w
        V_ = torch.cat([V_pad[:, :, i//3*self.d:i//3*self.d+h, i % 3*self.d:i % 3*self.d+w] for i in range(9)], dim=1)  # b, 9v, h, w
        nomask_ = torch.cat([nomask_pad[:, :, i//3*self.d:i//3*self.d+h, i % 3*self.d:i % 3*self.d+w] for i in range(9)], dim=1)  # b, 9, h, w

        Q = Q.permute(0,2,3,1).contiguous().view(b*h*w, self.d_k, 1)
        K = K_.permute(0,2,3,1).contiguous().view(b*h*w, 9, self.d_k)
        V = V_.permute(0,2,3,1).contiguous().view(b*h*w, 9, self.d_v).permute(0, 2, 1)    # bhw, v, 9
        nomask = nomask_.permute(0,2,3,1).contiguous().view(b*h*w, 9, 1)

        inner = torch.bmm(K, Q) #   b*h*w, 9, 1
        inner = inner * c**-0.5
        inner = inner.masked_fill(nomask.byte(), -np.inf)

        P = F.softmax(inner, dim=1)
        P = torch.where(torch.isnan(P), torch.full_like(P, 0), P)

        atten = torch.bmm(V, P).view(-1, self.d_v)  # b h w, v, 1
        atten = atten.view(b, h, w, self.d_v).permute(0, 3, 1, 2).contiguous().view(b//self.n_head, self.d_v*self.n_head, h, w)
        out = self.fc(atten)
        out = self.layer_norm(out + residual)
        return out  #   b, d_model, h, w

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None, last=False):
        if last:
            attn = torch.bmm(q, k.transpose(1, 2))
            # sigmoid
            return attn
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)

        attn = self.softmax(attn)   # sigmoid
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)

        return output, attn


def make_position_encoding(xp, batch, length, n_units, f=10000.):
    assert(n_units % 2 == 0)
    position_block = xp.broadcast_to(
        xp.arange(length)[None, None, :],
        (batch, n_units // 2, length)).astype('f')
    unit_block = xp.broadcast_to(
        xp.arange(n_units // 2)[None, :, None],
        (batch, n_units // 2, length)).astype('f')
    rad_block = position_block / (f * 1.) ** (unit_block / (n_units // 2))
    sin_block = xp.sin(rad_block)
    cos_block = xp.cos(rad_block)
    emb_block = xp.concatenate([sin_block, cos_block], axis=1)
    return emb_block

from torch.nn import InstanceNorm2d as In

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

class ListModule(nn.Module):

    def __init__(self, *args):
        super(ListModule, self).__init__()

        idx = 0
        for module in args:
            self.add_module(str(idx), module)
            idx += 1

    def __getitem__(self, idx):

        if idx < 0 or idx >= len(self._modules):
            raise IndexError('index {} is out of range'.format(idx))

        it = iter(self._modules.values())
        for i in range(idx):
            next(it)

        return next(it)

    def __iter__(self):
        return iter(self._modules.values())

    def __len__(self):
        return len(self._modules)


class AttentionLayer(nn.Module):
        def __init__(self, channel, reduction=2, multiply=True):
            super(AttentionLayer, self).__init__()
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Sequential(
                    nn.Linear(channel, channel // reduction),
                    nn.ReLU(inplace=True),
                    nn.Linear(channel // reduction, channel),
                    nn.Sigmoid()
                    )
            self.multiply = multiply
        def forward(self, x):
            b, c, _, _ = x.size()
            y = self.avg_pool(x).view(b, c)
            y = self.fc(y).view(b, c, 1, 1)
            if self.multiply == True:
                return x * y
            else:
                return y

class ChannelAttentionLayer(nn.Module):
    def __init__(self, d_model, d_h=None, reduction=2, multiply=True):
        super(ChannelAttentionLayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.l_b = nn.Linear(d_model, d_model//reduction)
        self.l_h = nn.Linear(d_model, d_model//reduction, bias=False)
        self.channel_fc = nn.Sequential(
            nn.Tanh(),
            nn.Linear(d_model//reduction, d_model),
            nn.Softmax(dim=1)
        )
        self.bn = nn.BatchNorm2d(d_model)

        self.d_model = d_model
        self.multiply = multiply

    def forward(self, Base, y_, h_t=None):
        _y = 1 - y_
        b, c, h, w = Base.size()
        base = self.avg_pool(Base * y_).view(b, c)
        base = self.l_b(base)
        # base = base.view(b, base.shape[1])
        if h_t is not None:
            h = self.l_h(h_t)
            base = base + h
        alpha = self.channel_fc(base).view(b, c, 1, 1) * self.d_model
        if self.multiply == True:
            paste = self.bn(Base * alpha)
            paste = paste # * y_
            Base = Base + paste
            return Base
        else:
            return alpha

from MobileNetDenseASPP import InvertedResidual
class SpatialAttentionLayer(nn.Module):
    '''
    Input:
    y_: 010, mask
    Base: b,c,h,w
    h_t: b, d_v
    Out: b,c,h,w
    '''
    def __init__(self, d_model, reduction=2, d_h=None, multiply=True):
        super(SpatialAttentionLayer, self).__init__()
        # self.l_s = nn.ModuleList([
        #     InvertedResidual(d_model, d_model, stride=1, expand_ratio=2, dilation=1),
        #     InvertedResidual(d_model, d_model, stride=1, expand_ratio=2, dilation=1)]
        # )
        # self.l_0 = nn.Sequential(
        #     nn.Conv2d(d_model, d_model, 3, 1, 1, groups=d_model),
        #     nn.BatchNorm2d(d_model),
        #     nn.ReLU()
        # )
        self.l_v = nn.Conv2d(d_model, d_model//reduction, 1, 1)
        self.l_h = nn.Linear(d_model, d_model//reduction, bias=False)
        self.spatial_fc = nn.Sequential(
            nn.Tanh(),
            nn.Conv2d(d_model//reduction, 1, 1, 1)
        )
        self.bn = nn.BatchNorm2d(d_model)
        self.multiply = multiply
    def forward(self, Base, y_, h_t=None, use_sigmoid=False, decoder = False):
        # inp = Base
        # for m in self.l_s:
        #     o = m(inp)
        #     inp = inp + o
        # return inp

        b, c, h, w = Base.size()
        if type(y_) != int:
            _y = (1-y_).byte()
        # Base = self.l_0(Base*y_)
        base = self.l_v(Base*y_)
        if h_t is None:
            Base_mask = Base * y_
            h_t = torch.mean(Base_mask.view(b, c, -1), dim=2)
        h_t = self.l_h(h_t)
        h_t = h_t.view(b, h_t.shape[1], 1, 1)
        base += h_t
        beta = self.spatial_fc(base)   # b, 1, h, w
        if use_sigmoid:
            beta = torch.sigmoid(beta).view(b, 1, h, w)
        else:
            if type(y_) != int and not decoder:
                beta = beta.masked_fill(_y, -np.inf)
                beta = beta.view(b, 1, -1)
                y_sum = torch.sum(y_, dim=[1,2,3], keepdim=True)
            else:
                y_sum = h * w
            beta = torch.softmax(beta, dim=2).view(b, 1, h, w) * y_sum   # b, d_model, h, w

        if torch.isnan(beta).sum() != 0:
            a = 1

        if self.multiply:
            paste = self.bn(Base * beta)
            paste = paste * y_
            Base = Base + paste
            return Base
        else:
            return beta
from torch.nn.modules import Module

from torch.nn.parameter import Parameter
import torch.nn.functional as F

class maskBN(Module):
    _version = 2

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(maskBN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        if self.affine:
            self.weight = Parameter(torch.Tensor(num_features))
            self.bias = Parameter(torch.Tensor(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features))
            self.register_buffer('running_var', torch.ones(num_features))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)
            self.register_parameter('num_batches_tracked', None)
        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)
            self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            self.weight.data.uniform_()
            self.bias.data.zero_()

    def forward(self, input, mask):
        b,c,h,w = input.shape
        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            self.num_batches_tracked += 1
            if self.momentum is None:  # use cumulative moving average
                exponential_average_factor = 1.0 / self.num_batches_tracked.item()
            else:  # use exponential moving average
                exponential_average_factor = self.momentum
        if self.training or not self.track_running_stats:
            mask_mean = torch.sum(mask.view(b, -1), dim=1)+1
            input = input.permute(0,2,3,1).contiguous().view(b,-1, c)
            mask = mask.permute(0,2,3,1).contiguous().view(b,-1, c)
            mean = torch.mean(torch.sum(input*mask,dim=1)/mask_mean[:,None],dim=0)
            input_sq = torch.pow(input-mean[None,None,:], 2)
            var = torch.mean(torch.sum(input_sq*mask,dim=1)/mask_mean[:,None],dim=0)
            self.running_mean = self.running_mean *exponential_average_factor + (1-exponential_average_factor) * mean
            self.running_var = self.running_var * exponential_average_factor + (1-exponential_average_factor) * var
            input = input.view(b,h,w,c).permute(0,3,1,2).contiguous()
            out = (input - mean)/torch.pow(var+self.eps, 0.5) * self.weight + self.bias
        else:
            out = (input - self.running_mean)/torch.pow(self.running_var+self.eps, 0.5) * self.weight + self.bias
        return out


    def extra_repr(self):
        return '{num_features}, eps={eps}, momentum={momentum}, affine={affine}, ' \
               'track_running_stats={track_running_stats}'.format(**self.__dict__)

    def _load_from_state_dict(self, state_dict, prefix, metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        version = metadata.get('version', None)

        if (version is None or version < 2) and self.track_running_stats:
            # at version 2: added num_batches_tracked buffer
            #               this should have a default value of 0
            num_batches_tracked_key = prefix + 'num_batches_tracked'
            if num_batches_tracked_key not in state_dict:
                state_dict[num_batches_tracked_key] = torch.tensor(0, dtype=torch.long)

        super(maskBN, self)._load_from_state_dict(
            state_dict, prefix, metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

class HardAttentionLayer(nn.Module):
    def __init__(self, d_model, d_k, d_h, random_thred=0.2, reduction = 2):
        super(HardAttentionLayer, self).__init__()
        self.l1 = nn.Conv2d(d_model, d_k, 1, 1)
        self.l2 = nn.Linear(d_model, d_k, bias=False)
        self.attend_fc = nn.Sequential(
            nn.Tanh(),
            nn.Conv2d(d_k, 1, 3, 1, 1),
            # nn.BatchNorm2d(1)
        )
        # self.attend_pn = nn.Sequential(
        #     nn.Conv2d(d_model, d_k, 1, 1),
        #     nn.BatchNorm2d(d_k),
        #     nn.Tanh(),
        #     nn.Conv2d(d_k, 2, 3, 1, 1),
        #     nn.Softmax(1)
        # )
        self.bn = maskBN(1)
    def forward(self, S, sem_seg, ins_seg, h_t=None):
        b, n, h, w = ins_seg.size()
        _ins_seg = (1 - ins_seg).byte()
        S = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)(S)
        # S = S*y_
        e_t_org = self.l1(S)
        # if h_t is not None:
        #     h_t = torch.mean(S.view(b, c, -1), dim=2)
        #     h_t = self.l2(h_t)
        #     e_t = e_128t + h_t.view(b, h_t.shape[1], 1, 1)
        e_t_org = self.attend_fc(e_t_org)
        e_t_org = self.bn(e_t_org, sem_seg)
        # p_n = self.attend_pn(S)
        # p_n = p_n[:,1,:,:].unsqueeze(1)
        e_t_org = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)(e_t_org) * sem_seg
        # e_t_out = e_t_out.detach()

        e_t = e_t_org.expand(-1, n, -1, -1).masked_fill(_ins_seg, -np.inf).view(b, n, -1)
        e_t = torch.softmax(e_t, dim=2).view(b, n, h, w)


        # e_t = e_t * ins_seg.float()
        # e_t = torch.softmax(e_t.view(b, n, -1), dim=2).view(b, n, h, w)
        # e_t = e_t.expand(-1, n, -1, -1).masked_fill(_y, -np.inf).view(b, n, -1)
        e_t_split = torch.where(torch.isnan(e_t), torch.full_like(e_t, 0).cuda(), e_t).view(b, n, h, w)
        # e_t = torch.sum(e_t_split, dim=1, keepdim=True)


        # alpha_t = torch.softmax(e_t, dim=1)
        # s_t = torch.multinomial(alpha_t, 1).long()
        # s_t = torch.zeros(b, h*w).scatter_(1, s_t, 1).view(b, 1, h, w)
        # alpha_t = alpha_t.view(b, h, w)
        return e_t_split, e_t_org#, p_n

class MobileV1ASPP(nn.Module):
    def __init__(self, inp, oup, stride, dilation=1, expand_ratio = 2, with_relu=False):
        super(MobileV1ASPP, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        self.use_res_connect = self.stride == 1 and inp == oup

        self.conv = nn.Sequential(
            # # pw
            nn.Conv2d(inp, inp * expand_ratio, 1, 1, 0, bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU6(inplace=True),
            # dw
            nn.Conv2d(inp * expand_ratio, inp * expand_ratio, kernel_size=3, stride=stride, padding=dilation,
                      dilation=dilation, groups=inp * expand_ratio, bias=False),
            nn.BatchNorm2d(inp* expand_ratio),
            nn.ReLU6(inplace=True),
            # pw-linear
            nn.Conv2d(inp * expand_ratio, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        )
        if with_relu:
            self.conv.add_module('relul', nn.ReLU6(inplace=True))

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class L0Layer(nn.Module):
    def __init__(self, d_model, reduction=2, is_last=False):
        super(L0Layer, self).__init__()
        self.d_model = d_model
        self.is_last = is_last
        # self.avg_pool = nn.AdaptiveAvgPool2d(1)
        out_channel = 2 # if is_last else 1
        self.l_i = nn.Conv2d(d_model, d_model//reduction, 3, 1, padding=1)
        self.last_fc = nn.Sequential(
            nn.LeakyReLU(),
            nn.Conv2d(d_model//reduction, out_channel, 3, 1, padding=1)
        )
        # if not is_last:
        #     self.l_N = nn.Sequential(
        #         nn.LeakyReLU(negative_slope=0.1, inplace=True)
        #     )
        # self.l_h = nn.Linear(d_model, d_model//reduction, bias=False)

        # self.sigma = Parameter(torch.Tensor(1))
        # self.sigma.data.fill_(d_model)
        # self.wp = Parameter(torch.Tensor(1))
        # self.wp.data.fill_(0.1)

        # self.sigma_fc = nn.Sequential(
        #     nn.Linear(d_model, d_model//reduction),
        #     nn.Tanh(),
        #     nn.Linear(d_model//reduction, 1),
        #     nn.Sigmoid()
        # )

        # self.wp_fc = nn.Sequential(
        #     nn.Linear(d_model, d_model//reduction),
        #     nn.Tanh(),
        #     nn.Linear(d_model//reduction, 1),
        #     nn.LeakyReLU(negative_slope=0.01)
        # )

    def forward(self, input, mask=None, h_t=None, use_sigmoid=False):
        '''

        :param input:
        :param mask:
        :param h_t:
        :return:
        '''
        # sigma = self.sigma_fc(h_t)
        # fi_map = CalPosition(input, point)
        # input = torch.cat([input, fi_map], dim=1)
        # input[:, :2, :, :] = input[:,:2,:,:] # + 2*sigma[:,:,np.newaxis, np.newaxis]*fi_map
        # sigma = self.sigma_fc(h_t)
        # wp = self.wp_fc(h_t)
        # log_K = - torch.norm(fi_map-h_t[:,:,np.newaxis, np.newaxis], dim=1, keepdim=True)/(self.d_model*2*sigma.view(sigma.shape[0],1,1,1))

        input = self.l_i(input)
        if h_t is None:
            h_t = 0
        else:
            h_t = self.l_h(h_t)
            h_t = h_t.view(h_t.shape[0], h_t.shape[1], 1, 1)

        s = self.last_fc(input + h_t)
        if False: # not self.is_last:
            b, c, h, w = s.shape
            s = s.view(b, c, -1)
            s = torch.sigmoid(s)
            # s = (s+1)/2
            # s = self.l_N(s)
            s_normal = torch.softmax(s, dim=2)
            # s_N = self.l_N(torch.mean(s, dim=2)).unsqueeze(-1) + 1
            # s = s_N * s_normal
            s = s.view(b, c, h, w)
        if mask is None:
            mask = 1

        if use_sigmoid:
            s = torch.sigmoid(s)
        if mask is not None:
            s = s * mask
        return s
        # s = torch.sigmoid(s + 0.1*wp[:, np.newaxis, np.newaxis]*log_K)*mask

class Embedding(nn.Module):
    def __init__(self, d_model, reduction=2):
        super(Embedding, self).__init__()
        self.sigma = nn.Sequential(
            nn.Linear(d_model, d_model//reduction),
            nn.Tanh(),
            nn.Linear(d_model//reduction, 1),
            nn.Sigmoid()
        )
        self.pad = nn.ConstantPad1d((0, d_model-2), 0)
    def forward(self, o_map, point, h):
        with torch.no_grad():
            fi_map = CalPosition(o_map, point)
            fi_map = self.pad(fi_map.permute(0,2,3,1)).permute(0,3,1,2)
        sigma = self.sigma(h).view(-1,1,1,1)
        o_map = o_map + fi_map * sigma * 2
        return o_map

def CalPosition(o_map, point):
    '''

    :param o_map: b, c, h, w
    :param point: a 2 len tuple of (rowlist, columnlist) of attend points
    :return:
    '''
    batch, c, h, w = o_map.shape

    # for b in range(batch):
    #     #add row
    #     o_map[b,1,:,:] = o_map[b,1,:,:] + torch.abs(position[0] - point[0][b])  # - o_map[b,1,point[0][b], point[1][b]]
    #     #add column
    #     o_map[b,0,:,:] = o_map[b,0,:,:] + torch.abs(position[1] - point[1][b])  # - o_map[b,1,point[0][b], point[1][b]]
    # return o_map

    position_r = torch.cat([torch.abs(position[0]-point[0][b]).unsqueeze(0) for b in range(batch)])
    position_c = torch.cat([torch.abs(position[1]-point[1][b]).unsqueeze(0) for b in range(batch)])
    position_all = torch.cat([position_c.unsqueeze(1), position_r.unsqueeze(1)], dim=1)
    return position_all

class UpDecoderLayer(nn.Module):
    def __init__(self, in_ch, out_ch, multi=1, loop=1, factor=1, use_mask=True, is_first=False, is_last=False, reduction1=2, reduction2=2):
        super(UpDecoderLayer, self).__init__()
        self.UpAtten = UpAttenLayer(in_ch, out_ch, reduction1, reduction2, multi, loop, factor, use_mask, is_first, is_last)
        if config.use_pyramid:
            self.pred = L0Layer(out_ch, 2, is_last)
        self.use_mask = use_mask
    def bin(self, i, j, N):
        i = bin(i)[2:]
        i = (N-len(i))*'0' + i
        j = bin(j)[2:]
        j = (N-len(j))*'0' + j
        out = list(map(int, i+j))
        return out
    def resize_p(self, points, factor):
        points_exp = [[n//factor for n in axis] for axis in points]
        points_2 = [[n%factor for n in axis] for axis in points]
        N = int(math.log(factor, 2))
        points_2 = [ self.bin(i,j, N) for i, j in zip(*points_2)]
        points_exp.append(points_2)
        return points_exp

    def resize(self, points, mask, mask_pre, factor, selection, training):
        points = self.resize_p(points, factor)
        mask, pro, mask_all = mask
        mask_all = nn.MaxPool2d(kernel_size=factor, stride=factor)(mask_all)
        # mask_all = nn.AvgPool2d(kernel_size=factor, stride=factor)(mask_all)

        # mask = nn.functional.interpolate(mask, (mask.shape[2]//factor, mask.shape[3]//factor), mode='bilinear', align_corners=False)
        pro = nn.AvgPool2d(kernel_size=factor, stride=factor)(pro)
        mask = nn.MaxPool2d(kernel_size=factor, stride=factor)(mask)
        if selection:
            if True: # random.random() < config.selection_threshold or not training:
                mask_pre = mask_pre[1]
                if True:
                    pass
                    # mask_pre = torch.softmax(mask_pre, dim=1)
                    # mask_pre = torch.split(mask_pre, [1,1], dim=1)[1]
                    # mask_pre = mask_pre.float().unsqueeze(1)
                else:
                    background = nn.MaxPool2d(kernel_size=2, stride=2)(mask_all)
                    mask_pre = torch.softmax(mask_pre, dim=1)[:,1,:,:].unsqueeze(1) * background
            else:
                mask_pre = mask_pre[0]
            # mask_pre = nn.Upsample(scale_factor=2, mode='nearest')(mask_pre)
        else:
            mask_pre = 1
            # mask_pre = nn.MaxPool2d(kernel_size=(mask_pre.shape[2]//mask.shape[2]), stride=(mask_pre.shape[2]//mask.shape[2]))(mask_pre)
            # mask_pre = nn.AvgPool2d(kernel_size=(mask_pre.shape[2]//mask.shape[2]), stride=(mask_pre.shape[2]//mask.shape[2]))(mask_pre)
            # mask_pre = nn.functional.interpolate(mask_pre, (mask.shape[2], mask.shape[3]), mode='bilinear', align_corners=False)

        return points, mask, pro, mask_all, mask_pre

    def forward(self, x1, x2, points, mask=None, mask_pre=None, selection=True, training=True):
        '''

        :param x1:
        :param x2:
        :param points:
        :param mask: 此处输出pred对应的真实掩码
        :param mask_pre: 低分辨率，前一个预测得到的掩码，用作注意力模型
        :param reduction:
        :return:
        '''

        # if self.use_decoder:
        # h_ = select(x, points)
        if config.use_pyramid:

            points, mask, pro, mask_all, mask_pre = self.resize(points, mask, mask_pre, config.H // x2.shape[2],
                                                           selection, training)
            if self.use_mask:
                x = self.UpAtten(x1, x2, points, mask_pre, mask_all, training)
            else:
                x = self.UpAtten(x1, x2, points, mask_pre, training=training)
            pred_out = self.pred(x)#, mask_all)
            return x, pred_out, mask
        else:
            factor = config.H // x2.shape[2]
            points = self.resize_p(points, factor)
            # mask = nn.AvgPool2d(kernel_size=factor, stride=factor)(mask)
            mask = nn.MaxPool2d(kernel_size=factor, stride=factor)(mask)
            x = self.UpAtten(x1, x2, points, mask)
            return x
        # else:
        #     h_ = select(x2, points)
        #     pred = self.pred(x2, mask_pre, h_)
        #     return pred, mask

class NonLocalLayer(nn.Module):
    def __init__(self, in_ch, out_ch, reduction=1):
        super(NonLocalLayer, self).__init__()
        self.g_net = nn.Conv2d(in_ch, out_ch, 1, 1)
        # if config.nonLocalType in ['Embedded Gaussian', 'Dot', 'Concatenation']:
        self.sita = nn.Linear(in_ch, in_ch//reduction)
        self.fi = nn.Conv2d(in_ch, in_ch//reduction, 1, 1)
        if config.nonLocalType == 'Concatenation':
            self.F = nn.Sequential(
                nn.Conv2d(in_ch*2, 1, 1, 1),
                nn.ReLU()
            )

    def forward(self, map, x):
        g = self.g_net(map)
        i = self.sita(x)
        js = self.fi(map)
        ##
        if config.nonLocalType in ['Dot', 'Embedded Gaussian']:
            i = i.unsqueeze(1)
            b,c,h,w = js.shape
            js = js.view(b, c, -1)
            f = torch.bmm(i, js)
            if config.nonLocalType == 'Embedded Gaussian':
                f = torch.exp(f)
            f = f.view(b, 1, h, w)
        elif config.nonLocalType == 'Concatenation':
            b, c, h,w = js.shape
            i = i.view(b, i.shape[1], 1, 1).expand(b, i.shape[1], h, w)
            conc = torch.cat([i, js], dim=1)
            f = self.F(conc)
        out = f * g + map
        return out

def Conv2d(inp, oup, kernel=1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, 1, kernel//2, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )

class UpAttenLayer(nn.Module):
    def __init__(self, in_ch, out_ch, reduction1=2, reduction2=2, multi=1, loop=1, factor=1, use_mask=True, is_first = False, is_last=False):
        super(UpAttenLayer, self).__init__()
        # if config.type == 0:
        #     self.up = nn.ConvTranspose2d(in_ch//reduction1, in_ch//reduction2, 2, stride=2)
        #     self.conv_1 = nn.Sequential(
        #         nn.Conv2d(in_ch, out_ch, 3, padding=1),
        #         nn.BatchNorm2d(out_ch),
        #         nn.ReLU(inplace=True)
        #     )
        #     self.conv_2 = nn.Sequential(
        #         nn.Conv2d(out_ch, out_ch, 3, padding=1),
        #         nn.BatchNorm2d(out_ch),
        #         nn.ReLU(inplace=True)
        #     )
        #     self.ch_1 = ChannelAttentionLayer(in_ch)
        #     self.sp_1 = SpatialAttentionLayer(out_ch)
        # elif config.type == 1:
        self.is_first = is_first
        N = 2*int(math.log(factor, 2)) if config.positionType else 0
        if use_mask:
            N += 2
        # up_ch = in_ch if self.is_first else in_ch // 2
        if config.use_pyramid:
            if is_first:
                pass
                # self.adj = MobileV1ASPP(in_ch, in_ch, 1, with_relu=False)
                # self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=(3, 3), stride=1, padding=1)
            else:
                self.up = nn.ConvTranspose2d(in_ch[1], out_ch, 2, stride=2)

            self.cross = nn.Sequential()
            # if is_last:
            #     self.cross.add_module('inconv', InvertedResidual(3, up_ch, 1, 2, 1))
            self.cross.add_module('up_feature', nn.Sequential(
                InvertedResidual(in_ch[0], out_ch, 1, 2, 1),
                # nn.Dropout2d(config.drop_rate, inplace=True),
                # InvertedResidual(up_ch, out_ch, 1, 2, 1),
                nn.Dropout2d(config.drop_rate, inplace=True),
                InvertedResidual(out_ch, out_ch - N, 1, 2, 1),
                # InvertedResidual(out_ch, out_ch-N-1, 1, 2, 3),
            )
                                  )

            conv_in = out_ch+out_ch if not is_first else out_ch
            self.conv1 = conv_1x1_bn(conv_in, out_ch)

        else:
            if is_first:
                self.adj = MobileV1ASPP(in_ch, out_ch*2, 1, with_relu=False)
            else:
                self.up = nn.ConvTranspose2d(out_ch*multi, out_ch, 2, stride=2)
                self.cross = nn.Sequential()
                for l in range(loop-1):
                    self.cross.add_module('cross_%s' % (l),
                                             nn.Sequential(
                                                 nn.Dropout2d(config.drop_rate, inplace=True),
                                                 InvertedResidual(out_ch, out_ch, 1, 2, 1),
                                                 nn.Dropout2d(config.drop_rate, inplace=True),
                                                 InvertedResidual(out_ch, out_ch, 1, 2, 2),
                                                 # InvertedResidual(out_ch, out_ch, 1, 2, 5),
                                             )
                                        )
            if config.positionType == 1:
                # self.conv1 = Conv2d(out_ch * 2 + factor**2, out_ch, 3)
                self.conv1 = conv_bn(out_ch*2 + N+1, out_ch)
            elif config.positionType == 0:
                self.conv1 = conv_bn(out_ch*2 + 1, out_ch)
        # self.ch_1 = ChannelAttentionLayer(in_ch)
        # self.sp_1 = SpatialAttentionLayer(in_ch)
        # self.globalAttention = NonLocalLayer(out_ch, out_ch)
        self.dilation_part1 = nn.Sequential(
                InvertedResidual(out_ch, out_ch, 1, 2, 1),
                InvertedResidual(out_ch, out_ch, 1, 2, 1)
        )

        self.dilation_part2 = nn.Sequential(
                InvertedResidual(out_ch, out_ch, 1, 2, 1),
                InvertedResidual(out_ch, out_ch, 1, 2, 1),
        )

    def conPosition(self, x, points):
        batch, _, h, w = x.shape
        if config.positionType == 1:
            N = int(math.log((config.H//h),2))
            position = torch.zeros(batch, N*2+1, h, w).cuda()
            for b in range(batch):
                position[b, -1, points[0][b], points[1][b]] = 1
                for t in range(2*N):
                    position[b, t, points[0][b], points[1][b]] = points[2][b][t]


        elif config.positionType == 0:
            c, pow = 1, 0
            position = torch.zeros(batch, 1, h, w).cuda()
            for b in range(batch):
                position[b, 0, points[0][b], points[1][b]] = 1
            ##
        x = torch.cat([x, position], dim=1)
        return x

    def Mask(self, x, mask=None):
        # mask_ = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)(mask)
        # mask_ = nn.Upsample(scale_factor=x2.shape[2] // mask.shape[2])(mask)
        if type(mask) != int:
            mask = nn.functional.interpolate(mask, (x.shape[2], x.shape[3]), mode='bilinear', align_corners=False)
            if mask.shape[1] == 2:
                mask = torch.softmax(mask, dim=1)
                mask = mask[:, 1, :, :].unsqueeze(1)
        x = x * mask
        return x

    def forward(self, x1, x2, points, mask=None, mask_all=None, training=True):
        if self.is_first:
            # x = self.adj(x2)
            # if config.use_pyramid:
            #     x = torch.cat([x, mask], dim=1)
            x = self.cross(x2)

        else:
            # if config.use_pyramid:
            #     x1 = torch.cat([x1, mask], dim=1)
            x1 = self.up(x1)

            # input is CHW
            diffY = x2.size()[2] - x1.size()[2]
            diffX = x2.size()[3] - x1.size()[3]
            x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2))
            x1_mask = self.Mask(x1, mask)

            x2 = self.cross(x2)

            # if config.use_mask:
            # x1 = self.Mask(x1, mask)

            x = torch.cat([x2, x1_mask], dim=1)

        if mask_all is not None:
            x = torch.cat([x, mask_all], dim=1)

        # if config.drop_rate > 0:
        #     x = F.dropout2d(x, p=config.drop_rate, training=training)

        x = self.conPosition(x, points)
        x = self.conv1(x)

        # attenPoint = torch.sum(x * position, dim=[2, 3])
        # x = self.globalAttention(x, attenPoint)

        # if config.use_pyramid:
        #     mask = nn.Upsample(scale_factor=x.shape[2] // mask.shape[2])(mask)
        #     x = torch.cat([x, mask], dim=1)

        # ch = torch.sum(x*position, dim=[2,3])#select(x, points)
        # x = self.ch_1(x, mask, ch)
        # sp = torch.sum(x*position, dim=[2,3])#select(x, points)
        # x = self.sp_1(x, mask, sp, decoder=True)
        if config.drop_rate > 0:
            x = F.dropout2d(x, p=config.drop_rate, training=training)
        x = self.dilation_part1(x)
        if not self.is_first:
            x = x + x1
        if config.drop_rate > 0:
            x = F.dropout2d(x, p=config.drop_rate, training=training)
        x = self.dilation_part2(x)
        return x


def select(map, point):
    if type(point)==list:
        b, c, h, w = map.shape
        if type(point[0])==list:
            point = [r*h+c for r,c in point]
        map = map.permute(0,2,3,1).view(b, h*w, c)
        ans = torch.cat([map[i][point[i]].unsqueeze(0) for i in range(b)])
        if len(ans.shape)==1:
            ans = ans.view(b, -1)
    else:
        ans = torch.sum((map*point), dim=[2, 3])
    return ans

def addFeature(feature, others, chunks=2):
    feature_splited = torch.chunk(feature, chunks, dim=1)
    feature_added = torch.cat([torch.cat([f, others], dim=1) for f in feature_splited], dim = 1)
    return feature_added

if __name__ == '__main__':
    pass

