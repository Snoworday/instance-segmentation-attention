import torch.nn as nn
from torch.nn import functional as F
import torch
from modules.utils import Encoder, Decoder, make_position_encoding
from losses.multi_loss import FocalLoss, DiceLoss, MmdLoss
import numpy as np
import cv2
import config
class atteNet(nn.Module):

    def __init__(self, num_layers, asppList, d_model, d_k, d_inner, n_head=2):
        super(atteNet, self).__init__()
        assert num_layers == len(asppList)
        d_k = d_v = d_model//n_head
        self.parameters = (num_layers, asppList, config.d_model, config.d_k, config.d_v, config.d_inner, config.n_head)
        self.encoder = Encoder(*self.parameters)
        self.decoderpip = DecoderPipline(config.decoer_num_layers, config.d_model, config.d_inner, config.n_head, config.d_k, config.d_v)
        self.position = None

        # self.conv_1x1 = nn.Conv2d(input_dimList[-1], input_dimList[0], kernel_size=(1, 1), stride=(1, 1))
    def forward(self, input, mask, ins_seg, dqnModel, max_iter=None, is_training=True):
        if self.position is None:
            self.position = self.set_position_encoding(input)
        enc_output = self.encoder(input, mask)
        # enc_output = enc_output.view(enc_output.shape[0], enc_output.shape[1], -1).permute(0, 2, 1)
        atten_loss = self.decoderpip(input, self.position, ins_seg, mask, encode=enc_output, dqnModel=dqnModel, max_iter=max_iter, is_training=True)

        # atten = self.conv_1x1(atten)
        return atten_loss

    def set_position_encoding(self, ins_seg_out):
            b, n_units, h, w = ins_seg_out.shape
            h_vec = np.tile(make_position_encoding(np, 1, h, n_units // 2, f=10000.)[:, :, :, np.newaxis], (1, 1, 1, w))
            w_vec = np.tile(make_position_encoding(np, 1, w, n_units // 2, f=10000.)[:, :, np.newaxis, :], (1, 1, h, 1))
            vec = np.concatenate([h_vec, w_vec], axis=1)
            return torch.from_numpy(vec).cuda()
    @property
    def Parameters(self):
        return self.parameters

class DecoderPipline(nn.Module):
    def __init__(self, num_layers, d_model, d_inner, n_head, d_k, d_v):
        super(DecoderPipline, self).__init__()
        self.decoder = Decoder(num_layers, d_model, d_inner, n_head, d_k, d_v)
        self.criterion_focal = FocalLoss(gamma=config.FocalLoss_gamma)
        self.criterion_dice = DiceLoss()
        self.criterion_mmd = MmdLoss()
        self.threshold = 16

    def map2list(self, input, ins_seg):
        '''input: b, 512, 256, 256
           ins_seg: b, 32, 256, 256
        '''
        b, c, h, w = input.shape
        n = ins_seg.shape[1]
        ins_seg_index = ins_seg.view(b, n, h*w)
        input = input.view(b, c, h*w)
        return input, ins_seg_index

    def removeSeled(self, mask, pred):
        pred_sum = torch.sum(mask * (pred>0.5).float(), dim=1)
        mask = mask - mask * (pred>0.5).float()
        return mask, pred_sum

    def getPred(self, index, ins_seg):
        '''b, 1; b, 32, 256*256'''
        '''b, 256*256'''
        batch, c, l = ins_seg.shape
        # num_bakeup = index.shape[1]
        # ins_seg = ins_seg.unsqueeze(1).expand(-1, num_bakeup, -1, -1)
        # ins_seg = ins_seg.view(batch*num_bakeup, c, l)
        # index = index.view(-1)
        # out = torch.index_select(ins_seg, 1, index)
        # index = index.long()
        out = torch.cat([torch.index_select(ins_seg[i], 0, index[i].long()) for i in range(batch)])
        out = out.view(batch, -1).float()
        return out

    def Attenloss(self, pred, target, mask, loss_type='Multi'):
        pred = pred * mask
        target = target * mask
        if loss_type in ['Focal', 'Multi']:
            # mask = mask.view(mask.shape[0], -1) #   mask:no_grad
            focal_loss = self.criterion_focal(pred, target, mask)
        if loss_type in ['Dice', 'Multi']:
            dice_loss = self.criterion_dice(pred, target)
        # if loss_type in ['Mmd', 'Multi']:
        #     mmd_loss = self.criterion_mmd(pred, target)

        loss = config.FocalWeight * focal_loss + dice_loss # + mmd_loss
        pred_ = (pred > 0.5).float()
        U1 = target.sum(dim=1)
        U2 = pred_.sum(dim=1)
        I = (pred_*target).sum(dim=1)
        IOU = 2*I/(U1+U2)
        return loss, IOU

    def compress(self, done, *input):
        out = [i[done==0] for i in input]
        return out

    def onehot2index(self, t):
        return torch.from_numpy(np.where(t==0)[0]).cuda()

    def getSelected(self, input, actions):
        if len(input.shape)==4:
            batch, c, h, w = input.shape
            input = input.view(batch, c, h*w).permute(0, 2, 1)
        else:
            batch = input.shape[0]
        return torch.cat([input[b][actions[b]].unsqueeze(0) for b in range(batch)])


    def forward(self, input, position, ins_seg, mask, encode, dqnModel, max_iter, is_training=True):
        ''' input: type:list b, 512, 256, 256
            index: b,
            encode: b, 256*256, 512
            mask: b, h*w
        '''
        # position embedding should be more fancy
        input = input # + position

        mask = mask.view(mask.shape[0], -1)
        mask_sum = torch.sum(mask, dim=1)
        input, ins_index = self.map2list(input, ins_seg) # input: b, 512, 256*256 ; input_list_index: b, class_num, 256*256
        input = input.permute(0, 2, 1)
        done = torch.zeros(mask.shape[0]).byte().cuda()    #  if所有输入都为0个
        loss = torch.zeros(done.shape[0]).cuda()
        # encode = torch.randn(2, 256*256, 24)
        if max_iter is not None:
            max_iter = max_iter.max()
        else:
            max_iter = config.max_iter
        iter = 0

        while iter < max_iter and done.min()==0:
            with torch.no_grad():
                candidate, candidate_idx, actions = dqnModel.act(encode, ins_index, input, mask)    # candidate(tensor):b, 512; b, 1
            pred = self.decoder(self.getSelected(input, actions), encode, mask)                 # PRED(list):b, 256*256,[0,1]，为sigmoid之后的值
            # pred = self.decoder(candidate, encode, mask)
            gold = self.getPred(candidate_idx, ins_index)
            mask_, pred_sum = self.removeSeled(mask, gold)  # b, 256*256

            loss_atten, iou = self.Attenloss(pred, gold, mask)      #loss_attn: undone中1数量的长度
            loss_atten = loss_atten * pred_sum  # (pred_sum + 1)

            loss = loss.scatter_add_(0, self.onehot2index(done), loss_atten)  # loss始终是batch长度
            done_selected = (mask_.sum(dim=1) == 0).cuda() * (torch.sum(mask_, dim=1).byte() + 1)
            done = done.scatter_(0, self.onehot2index(done), done_selected)    # have 0
            dqnModel.push(encode.detach(), actions.detach(), iou.detach(), mask.detach(), mask_.detach(), (done!=0).detach())
            # if done_selected.max() != 0:
            input, ins_index, encode, mask = self.compress(done, input, ins_index, encode, mask_)
            mask = mask.detach()
            iter += 1
        done = torch.where(done == 0, torch.full_like(done, max_iter).cuda(), done)
        loss = loss / (mask_sum)    # - done.float())
        return loss


def prodTemplate(asppList, input_num, usegpu=True):
    n_temp = 9
    weights = []
    for d in asppList:
        side = n_temp * 2 + 3
        conv = torch.zeros([n_temp, side, side])
        for i in range(n_temp):
            x = i//3
            y = (i - x * 3) * d
            conv[i][x][y] = 1
        conv = conv.unsqueeze(1).expand(n_temp, input_num, 3, 3)
        conv = conv.contiguous()
        conv = conv.view(n_temp*input_num, 1, 3, 3)
        if usegpu:
            conv = conv.cuda()
        weights.append(conv)
    return weights



if __name__ == '__main__':
    input = torch.randn(2, 24, 256, 256)
    ins_seg = torch.where(input>0, torch.Tensor(input.shape).fill_(1), torch.Tensor(input.shape).fill_(0))[:, :16, :, :]
    print(ins_seg.shape)
    mask = torch.where(input>0, torch.Tensor(input.shape).fill_(1), torch.Tensor(input.shape).fill_(0))[:, 10, :, :]
    print(mask.shape)
    tmp = DecoderPipline(3, 24, 48, 2, 24, 24)
    a = tmp(input, ins_seg, mask)
    # l = _AttenAsppBlock(dilation_rate=3, input_dim=32, v_dim=16, f_dim=64)
    # x = torch.randn(2, 32, 256, 256)
    # out = l(x)
    # print(out.shape)
