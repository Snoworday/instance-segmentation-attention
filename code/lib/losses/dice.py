from torch.nn.modules.loss import  _Loss, _WeightedLoss
from torch.nn import functional as F
import torch
import numpy as np
import os
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from losses.multi_loss import FocalLoss
import math
def dice_coefficient(input, target, mask=None, smooth=1.0, time=2, map_weight=0):
    """input : is a torch variable of size BatchxnclassesxHxW representing
    log probabilities for each class
    target : is a 1-hot representation of the groundtruth, shoud have same size
    as the input"""

    assert input.size() == target.size(), 'Input sizes must be equal.'
    assert input.dim() == 4, 'Input must be a 4D Tensor.'
    uniques = np.unique(target.data.cpu().numpy())
    assert max(list(uniques)) <= 1, 'Target must only in [0, 1].'
    assert min(list(uniques)) >= 0, 'Target must only in [0, 1].'

    assert smooth > 0, 'Smooth must be greater than 0.'

    probs = F.softmax(input, dim=1)
    target_f = target.float()

    num = probs * target_f * (map_weight+1)         # b, c, h, w -- p*g
    if mask is not None:
        num = num * mask
    num = torch.sum(num, dim=3)    # b, c, h
    num = torch.sum(num, dim=2)    # b, c
    if time==1:
        den1 = probs * (map_weight+1)
    else:
        den1 = probs * probs * (map_weight+1)           # b, c, h, w -- p^2
    if mask is not None:
        den1 = den1 * mask
    den1 = torch.sum(den1, dim=3)  # b, c, h
    den1 = torch.sum(den1, dim=2)  # b, c
    if time==1:
        den2=target_f * (map_weight+1)
    else:
        den2 = target_f * target_f * (map_weight+1)     # b, c, h, w -- g^2
    if mask is not None:
        den2 = den2 * mask
    den2 = torch.sum(den2, dim=3)  # b, c, h
    den2 = torch.sum(den2, dim=2)  # b, c

    dice = (2 * num + smooth) / (den1 + den2 + smooth)

    return dice


def dice_loss(input, target, optimize_bg=False, weight=None,
              smooth=1.0, size_average=True, reduce=True, mask=None, time=2, map_weight=0):
    """input : is a torch variable of size BatchxnclassesxHxW representing
    log probabilities for each class
    target : is a 1-hot representation of the groundtruth, shoud have same size
    as the input

    weight (Variable, optional): a manual rescaling weight given to each
            class. If given, has to be a Variable of size "nclasses"""
    dice = dice_coefficient(input, target, mask, smooth=smooth, time=time, map_weight=map_weight)


    if not optimize_bg:
        # we ignore bg dice val, and take the fg
        dice = dice[:, 1:]

    if not isinstance(weight, type(None)):
        if not optimize_bg:
            weight = weight[1:]             # ignore bg weight
        weight = weight.size(0) * weight / weight.sum()  # normalize fg weights
        dice = dice * weight      # weighting

    # loss is calculated using mean over fg dice vals
    dice_loss = 1 - dice.mean(1)

    if not reduce:
        return dice_loss

    if size_average:
        return dice_loss.mean()

    return dice_loss.sum()
from torch import nn

def mmd_penalty(sample_qz, sample_pz, opts, kernel='IMQ'):
    m = sample_pz.shape[0]
    n = sample_qz.shape[0]
    if m<2 or n<2:
        return 0
    # print(m, n)
    half_size = (n**2 - n) / 2
    norms_pz = torch.sum(torch.pow(sample_pz, 2), dim=1, keepdim=True)
    dotprods_pz = torch.matmul(sample_pz, sample_pz.t())
    distances_pz = norms_pz + norms_pz.t() - 2 * dotprods_pz

    norms_qz = torch.sum(torch.pow(sample_qz, 2), dim=1, keepdim=True)
    dotprods_qz = torch.matmul(sample_qz, sample_qz.t())
    distances_qz = norms_qz + norms_qz.t() - 2 * dotprods_qz

    dotprods = torch.matmul(sample_qz, sample_pz.t())
    distances = norms_qz + norms_pz.t() - 2. * dotprods
    if kernel=='IMQ':
        sigma2_p = 1.0
        if opts['pz'] == 'normal':
            Cbase = 2. * opts['zdim'] * sigma2_p
        elif opts['pz'] == 'sphere':
            Cbase = 2.
        elif opts['pz'] == 'uniform':
            Cbase = opts['zdim']
        stat = 0.
        for scale in [.1, .2, .5, 1., 2., 5., 10.]:
            C = Cbase * scale
            res1 = torch.sum( C / (C + distances_qz) * (1-torch.eye(n).cuda()) / (n**2 - n))
            res1 += torch.sum( C / (C + distances_pz) * (1-torch.eye(m).cuda()) / (m**2 - m))
            res2 = C / (C + distances)
            res2 = torch.sum(res2) * 2. / (n * m)
            stat += res1 - res2
    return stat


def compute_interD(input, D='_intrin'):
    if D=='intrin':
        distances = - torch.matmul(input, input.t())
    else:
        pingfang = torch.sum(torch.pow(input, 2), dim=1, keepdim=True)
        jiaocha = torch.matmul(input, input.t())
        distances = pingfang + pingfang.t() - 2 * jiaocha
    return distances
def gl_loss(encode, decode, opt, weight=None, kernel='IMQ',
               smooth=1.0, size_average=True, reduce=True):
    b, _, _ = decode.shape
    decode = decode.reshape(b, -1)
    encode_d = compute_interD(encode)
    decode_d = compute_interD(decode)

    encode_d = encode_d.reshape(-1)
    decode_d = decode_d.reshape(-1)

    en_sort_idx = torch.sort(encode_d, -1, True)[1] #降序，每个值代表第idx大的元素对应的索引
    de_sort_idx = torch.sort(decode_d, -1, True)[1]
    en_idx = torch.FloatTensor(en_sort_idx.shape).cuda()
    de_idx = torch.FloatTensor(de_sort_idx.shape).cuda()
    for idx, (en, de) in enumerate(zip(*[en_sort_idx, de_sort_idx])):
        en_idx[en] = idx
        de_idx[de] = idx

    penalty_loss = torch.sum((de_idx - en_idx)*encode_d)/ (b**2 - b) / (64*34**0.5)

    return penalty_loss

def mmd_penalty_with_p(sample_qz, sample_pz, q_, p_, opts, kernel='RBF'):
    m = sample_pz.shape[0]
    n = sample_qz.shape[0]
    #   print('n', n)
    q_ = q_/torch.sum(q_)
    p_ = p_/torch.sum(p_)
    if m<2 or n<2:
        return 0
    # print(m, n)
    with torch.no_grad():
        norms_pz = torch.sum(torch.pow(sample_pz, 2), dim=1, keepdim=True)
        dotprods_pz = torch.matmul(sample_pz, sample_pz.t())
        distances_pz = norms_pz + norms_pz.t() - 2 * dotprods_pz

    norms_qz = torch.sum(torch.pow(sample_qz, 2), dim=1, keepdim=True)
    dotprods_qz = torch.matmul(sample_qz, sample_qz.t())
    distances_qz = norms_qz + norms_qz.t() - 2 * dotprods_qz

    dotprods = torch.matmul(sample_qz, sample_pz.t())
    distances = norms_qz + norms_pz.t() - 2. * dotprods
    if kernel=='IMQ':
        # print(q_.shape)
        if opts['pz'] == 'normal':
            Cbase = 2. * opts['zdim'] * sigma2_p
        elif opts['pz'] == 'sphere':
            Cbase = 2.
        elif opts['pz'] == 'uniform':
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
        sigma2_k = opts['sigma']
        res1 = torch.sum( torch.exp(distances_qz / -2./sigma2_k) * q_ * q_.t() )*0.5
        with torch.no_grad():
            res2 = torch.sum( torch.exp(distances_pz / -2./sigma2_k) * p_ * p_.t())*0.5
        res3 = torch.sum( torch.exp(distances / -2./ sigma2_k) * q_ * p_.t() )
        stat = res1 + res2 - res3
    return stat


def decoder_mmd_loss(input, target, opts, fixed):
    idx_h, idx_v = fixed
    batch, h, w = input.shape
    th1 = float(torch.mean(input)*h*w)/200
    th2 = float(torch.mean(target)*h*w)/200
    #   print(th)
    uniform_select = (torch.rand(input.shape)).detach().cuda()
    input_sel = torch.where(input > uniform_select*th1, torch.full_like(input, 1).cuda(), torch.full_like(input, 0).cuda()).byte()
    target_sel = torch.where(target > uniform_select*th2, torch.full_like(target, 1).cuda(), torch.full_like(target, 0).cuda()).byte()

    input_and_target = [[torch.cat([idx_v[input_sel[b]].unsqueeze(1), idx_h[input_sel[b]].unsqueeze(1)], 1),
                         torch.cat([idx_v[target_sel[b]].unsqueeze(1), idx_h[target_sel[b]].unsqueeze(1)], 1)] for b in
                        range(batch) if torch.sum(input_sel[b]) and torch.sum(target_sel[b])]
    input_and_target_p = [[input[b][input_sel[b]].unsqueeze(1),
                            target[b][target_sel[b]].unsqueeze(1) ] for b in
                        range(batch) if torch.sum(input_sel[b]) and torch.sum(target_sel[b])]
    position_loss = 0
    try:
        input_idx, target_idx = zip(*input_and_target)
        input_p, target_p = zip(*input_and_target_p)

        input_idx = [i[:300] for i in input_idx]
        target_idx= [i[:300] for i in target_idx]
        input_p = [i[:300] for i in input_p]
        target_p = [i[:300] for i in target_p]
        print(len(input_p[0]), len(input_p[1]))

        for _b in range(len(input_p)):
            # l += torch.sum(input_p[_b]*input_idx[_b]) + torch.sum(target_p[_b]*target_idx[_b])
            position_loss += mmd_penalty_with_p(input_idx[_b], target_idx[_b], input_p[_b], target_p[_b], opts)

    except:
        a=1
        pass
    # position_loss = torch.sum([mmd_penalty_with_p(input_idx[i], target_idx[i], input_p[i], target_p[i], opts) for i in range(len(input_idx))])

    # input_sum = torch.sum(input, dim=[1, 2])
    # target_sum = torch.sum(target, dim=[1, 2])
    # area_loss = torch.sum(torch.abs(input_sum-target_sum))/(batch*h*w)


    loss =  position_loss # + area_loss*100
    return loss

def c_loss(input, target, loss_func):
    loss = loss_func(input, target)
    return loss

class MatchLoss(_WeightedLoss):

    def __init__(self, decoder, decoder_lr_opt, usegpu, opt, load_decoder_model_path='', weight=None,
                 smooth=1.0, size_average=True, reduce=True):
        """input : is a torch variable of size BatchxnclassesxHxW representing
        log probabilities for each class
        target : is a 1-hot representation of the groundtruth, shoud have same
        size as the input

        weight (Variable, optional): a manual rescaling weight given to each
                class. If given, has to be a Variable of size "nclasses"""

        super(MatchLoss, self).__init__(weight, size_average)
        self.smooth = smooth
        self.class_weight = weight
        self.opt = opt
        self.usegpu = usegpu
        self.c_loss_function = FocalLoss(2)
        self.load_model_path = load_decoder_model_path
        self.fixed = None
        self.__define_decoder_and_optim(decoder, decoder_lr_opt)
        if usegpu:
            self.decoder.cuda()
            self.c_loss_function.cuda()
        self.all_loss = 0

    def __define_decoder_and_optim(self, decoder, lr_opt):
        self.decoder = decoder
        self.__load_weights()
        parameters = filter(lambda p: p.requires_grad,
                             self.decoder.parameters())
        self.optimizer = optim.Adam(parameters, lr_opt['learning_rate'], betas=(0.5, 0.999), weight_decay=lr_opt['weight_decay'])
        self.lr_scheduler = ReduceLROnPlateau(
            self.optimizer, mode='min', factor=lr_opt['lr_drop_factor'],
            patience=lr_opt['lr_drop_patience'], verbose=True)

    def step(self, cost):
        self.lr_scheduler.step(cost)

    def forward(self, sample_qz, ins_seg_annotations, sample_pz=None):
        # _assert_no_grad(ins_seg_annotations)
        # _assert_no_grad(sample_pz)
        ins_seg_predictions = self.decoder(sample_qz)   #input 获取model的输出，具体哪层要改
        # match qz & area

        #for 单调性
        penalty_loss = gl_loss(sample_qz, ins_seg_predictions, self.opt,
                         weight=self.weight, smooth=self.smooth)
        reconstruction_loss = c_loss(ins_seg_predictions, ins_seg_annotations, self.c_loss_function)
        if self.fixed is None:
            self.set_fixed(ins_seg_predictions)
        # area loss & position loss
        decoder_loss = decoder_mmd_loss(ins_seg_predictions, ins_seg_annotations, self.opt, self.fixed)

        self.all_loss = 100*reconstruction_loss + penalty_loss + self.opt['lambda'] * decoder_loss
        return self.all_loss, reconstruction_loss, penalty_loss, decoder_loss
    def optimize(self):

        torch.nn.utils.clip_grad_norm_(
            self.decoder.parameters(), 10)
        self.optimizer.step()
    def __load_weights(self):

        if self.load_model_path != '':
            assert os.path.isfile(self.load_model_path), 'Model : {} does not \
                exists!'.format(self.load_model_path)
            print('Loading model from {}'.format(self.load_model_path))

            model_state_dict = self.decoder.state_dict()

            if self.usegpu:
                pretrained_state_dict = torch.load(self.load_model_path)
            else:
                pretrained_state_dict = torch.load(
                    self.load_model_path, map_location=lambda storage,
                    loc: storage)

            model_state_dict.update(pretrained_state_dict)
            self.decoder.load_state_dict(model_state_dict)
    def save_weights(self, path):
        torch.save(self.decoder.state_dict(), path)
    def set_fixed(self, input):
        _, h, w = input.shape
        idx_h = torch.linspace(0, w-1, w).unsqueeze(0).repeat(h, 1).cuda()
        idx_v = idx_h.t()
        self.fixed = [idx_h, idx_v]

class DiceLoss(_WeightedLoss):

    def __init__(self, optimize_bg=False, weight=None,
                 smooth=1.0, size_average=True, reduce=True):
        """input : is a torch variable of size BatchxnclassesxHxW representing
        log probabilities for each class
        target : is a 1-hot representation of the groundtruth, shoud have same
        size as the input

        weight (Variable, optional): a manual rescaling weight given to each
                class. If given, has to be a Variable of size "nclasses"""

        super(DiceLoss, self).__init__(weight, size_average)
        self.optimize_bg = optimize_bg
        self.smooth = smooth
        self.reduce = reduce
        self.size_average = size_average

    def forward(self, input, target, mask=None, time=2, map_weight=0):
        # _assert_no_grad(target)
        return dice_loss(input, target, optimize_bg=self.optimize_bg,
                         weight=self.weight, smooth=self.smooth,
                         size_average=self.size_average,
                         reduce=self.reduce, mask=mask, time=time, map_weight=map_weight)


class DiceCoefficient(torch.nn.Module):

    def __init__(self, smooth=1.0):
        """input : is a torch variable of size BatchxnclassesxHxW representing
        log probabilities for each class
        target : is a 1-hot representation of the groundtruth, shoud have same
        size as the input"""
        super(DiceCoefficient, self).__init__()

        self.smooth = smooth

    def forward(self, input, target):
        # _assert_no_grad(target)
        return dice_coefficient(input, target, smooth=self.smooth)


if __name__ == '__main__':
    #input b,h,w ; target b, h, w
    from torch.autograd import Variable
    input = torch.FloatTensor([[-3, -1, 100, -20], [-5, -20, 5, 5]])
    input = Variable(input.unsqueeze(2).unsqueeze(3))
    target = torch.IntTensor([[0, 0, 1, 0], [0, 0, 0, 1]])
    target = Variable(target.unsqueeze(2).unsqueeze(3))

    weight = Variable(torch.FloatTensor(np.array([1.0, 1.0, 1.0, 1.0])))

    dice_loss_1 = DiceLoss(weight=weight)
    # dice_loss_2 = DiceLoss(size_average=False)
    # dice_loss_3 = DiceLoss(reduce=False)

    print( dice_loss_1(input, target))
    # print( dice_loss_2(input, target))
    # print( dice_loss_3(input, target))
    print( dice_coefficient(input, target, smooth=1.0))
