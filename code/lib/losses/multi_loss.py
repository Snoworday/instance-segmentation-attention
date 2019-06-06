from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import config

def one_hot(index, classes):
    size = index.size() + (classes,)
    view = index.size() + (1,)
    mask = torch.Tensor(*size).fill_(0)
    index = index.view(*view)
    ones = 1.
    if isinstance(index, Variable):
        ones = Variable(torch.Tensor(index.size()).fill_(1))
        mask = Variable(mask, volatile=index.volatile)
    return mask.scatter_(1, index, ones)


class FocalLoss(nn.Module):
    def __init__(self, gamma, num_classes=1):
        super(FocalLoss, self).__init__()
        self.num_classes = num_classes
        self.gamma = gamma

    def forward(self, preds, targets, alpha=0, mask=None, map_weight=0, isDealP=True):
        '''

        :param preds: L, 2
        :param targets: L
        :return:
        '''
        targets = targets.float()
        preds = torch.softmax(preds, dim=1)
        pt = preds.detach()
        preds = preds.clamp(1e-7, 1. - 1e-7)
        F_loss_1 = -1 * (1-alpha) * (1-pt[:,1]) ** self.gamma * torch.log(preds[:,1]) * targets * (map_weight+1)
        F_loss_0 = -1 * (1+alpha) * (1-pt[:,0])**self.gamma*torch.log(preds[:,0]) * (1 - targets) * (map_weight+1)
        # loss = torch.sum(F_loss_1 + F_loss_0, dim=1)
        loss = F_loss_1 + F_loss_0
        return loss

class BceLoss(nn.Module):
    def __init__(self):
        super(BceLoss, self).__init__()
        pass
    def forward(self, pred, target, mask):
        N = target.size(0)
        pred = pred.view(N, -1).clamp(1e-7, 1. - 1e-7)
        target = target.view(N, -1)
        mask = mask.view(N, -1)
        l = target * torch.log(pred) + (1-target) * torch.log(1 - pred)
        l = l * mask
        l = torch.sum(l, dim=1)
        return l

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, input, target):
        N = target.size(0)
        smooth = 1

        input_flat = input.view(N, -1)
        target_flat = target.view(N, -1)

        intersection = input_flat * target_flat
        background = torch.sum(target_flat, dim=1)

        loss = 2 * (intersection.sum(1) + smooth) / (input_flat.sum(1) + target_flat.sum(1) + smooth)
        loss = (1 - loss) * background

        return loss

class MmdLoss(nn.Module):
    def __init__(self, mmd_l = config.mmd_l):
        super(MmdLoss, self).__init__()
        self.Pool = None
        self.h, self.w = mmd_l, mmd_l
        self.fixed = self.set_fixed()

        self.maxp1 = nn.MaxPool2d(kernel_size=4)
        self.maxp2 = nn.MaxPool2d(kernel_size=4)
        self.avgp1 = nn.AvgPool2d(kernel_size=4)
        self.avgp2 = nn.AvgPool2d(kernel_size=4)

    def set_fixed(self):
        idx_h = torch.linspace(0, self.w - 1, self.w).unsqueeze(0).repeat(self.h, 1).cuda()
        idx_v = idx_h.t()
        return [idx_h, idx_v]

    def forward(self, input, target):
        batch, L = input.shape
        h = w = int(L**0.5)
        target = target.view(batch, 1, h, w)


        idx_h, idx_v = self.fixed
        th_input = max(float(torch.mean(input) * h * w) / 500, 0.01)
        th_target = max(float(torch.mean(target) * h * w) / 100, 0.01)
        uniform_select_input = (torch.rand(input.shape) * th_input).cuda()
        uniform_select_target= (torch.rand(target.shape)* th_target).cuda()
        input_sel = torch.where(input > uniform_select_input, torch.full_like(input, 1).cuda(),
                                torch.full_like(input, 0).cuda())
        target_sel = torch.where(target > uniform_select_target, torch.full_like(target, 1).cuda(),
                                 torch.full_like(target, 0).cuda())

        input_sel = self.maxp1(input_sel).squeeze(1).byte()
        target_sel = self.maxp2(target_sel).squeeze(1).byte()


        input_and_target = [[torch.cat([idx_v[input_sel[b]].unsqueeze(1), idx_h[input_sel[b]].unsqueeze(1)], 1),
                             torch.cat([idx_v[target_sel[b]].unsqueeze(1), idx_h[target_sel[b]].unsqueeze(1)], 1)] for b
                            in range(batch) if torch.sum(input_sel[b]) and torch.sum(target_sel[b])]
        try:
            input_idx, target_idx = zip(*input_and_target)
        except:
            return torch.zeros([input.shape[0]]).cuda()
        input = self.avgp1(input).squeeze(1)
        target = self.avgp2(target).squeeze(1)

        input_and_target_p = [[input[b][input_sel[b]].unsqueeze(1),
                               target[b][target_sel[b]].unsqueeze(1)] for b in
                              range(batch) if torch.sum(input_sel[b]) and torch.sum(target_sel[b])]
        input_p, target_p = zip(*input_and_target_p)
        # print('start')
        # a = [ i.shape[0] for i in input_idx  ]
        # print(sum(a))
        try:
            position_loss = torch.cat([mmd_penalty_with_p(input_idx[i], target_idx[i], input_p[i], target_p[i]) for i in
                                 range(len(input_idx))])
        except:
            a = 1
        # print('END')
        input_sum = torch.sum(input, dim=[1, 2])
        target_sum = torch.sum(target, dim=[1, 2])
        area_loss = torch.pow(input_sum - target_sum, 2)/ (h * w)

        loss = position_loss + area_loss
        return loss

def mmd_penalty_with_p(sample_qz, sample_pz, q_, p_, kernel='RBF'):
    m = sample_pz.shape[0]
    n = sample_qz.shape[0]
    #   print('n', n)
    q_ = q_/torch.sum(q_)
    p_ = p_/torch.sum(p_)
    if m<1 or n<1:
        return torch.zeros(1).xuda()
    # print(m, n)
    norms_pz = torch.sum(torch.pow(sample_pz, 2), dim=1, keepdim=True)
    dotprods_pz = torch.matmul(sample_pz, sample_pz.t())
    distances_pz = norms_pz + norms_pz.t() - 2 * dotprods_pz

    norms_qz = torch.sum(torch.pow(sample_qz, 2), dim=1, keepdim=True)
    dotprods_qz = torch.matmul(sample_qz, sample_qz.t())
    distances_qz = norms_qz + norms_qz.t() - 2 * dotprods_qz

    dotprods = torch.matmul(sample_qz, sample_pz.t())
    distances = norms_qz + norms_pz.t() - 2. * dotprods
    sigma2_p = 1
    if kernel=='IMQ':
        # print(q_.shape)
        if config.mmd_pz == 'normal':
            Cbase = 2. * opts['zdim'] * sigma2_p
        elif config.mmd_pz == 'sphere':
            Cbase = 2.
        elif config.mmd_pz == 'uniform':
            Cbase = opts['zdim']
        stat = 0.
        for scale in [.1, .2, .5, 1., 2., 5., 10.]:
            C = Cbase * scale
            res1 = torch.sum( q_ * q_.t() * C / (C + distances_qz))  # / (n**2 - n))
            res1 += torch.sum( p_ * C / (C + distances_pz) * p_.t())  # / (m**2 - m))
            res2 = torch.sum(q_ * C / (C + distances) * p_.t() * 2.)
            # res2 = torch.sum(res2) * 2. / (n * m)
            stat += res1 - res2
    elif kernel == 'RBF':
        sigma2_k = 64
        res1 = torch.sum( torch.exp(distances_qz / -2./sigma2_k) * q_ * q_.t() )*0.5
        res1 +=torch.sum( torch.exp(distances_pz / -2./sigma2_k) * p_ * p_.t())*0.5
        res2 = torch.sum( torch.exp(distances / -2./ sigma2_k) * q_ * p_.t() )
        stat = res1 - res2
        stat = stat.unsqueeze(0)
    return stat
