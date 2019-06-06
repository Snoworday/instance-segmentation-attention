from torch.nn.modules.loss import _Loss
from torch.autograd import Variable
import torch
import numpy as np


def calculate_means(pred, gt, n_objects, max_n_objects, usegpu, M='intri'):
    """pred: bs, height * width, n_filters
       gt: bs, height * width, n_instances"""

    bs, n_loc, n_filters = pred.size()
    n_instances = gt.size(2)

    pred_repeated = pred.unsqueeze(2).expand(
        bs, n_loc, n_instances, n_filters)  # bs, n_loc, n_instances, n_filters
    # bs, n_loc, n_instances, 1
    gt_expanded = gt.unsqueeze(3)

    # randsel = torch.randn(gt.shape).cuda() * gt
    # randsel = randsel / torch.sum(randsel, dim=1, keepdim=True)
    # randsel = randsel.unsqueeze(3)
    # rand_masked = randsel * pred_repeated
    pred_masked = pred_repeated * gt_expanded

    # pred_np = pred.cpu().detach().numpy()
    # gt_np   = gt.cpu().detach().numpy()
    # np.save('pred.npy', pred_np)
    # np.save('gt.npy', gt_np)

    means = []
    norms = []
    for i in range(bs):
        _n_objects_sample = n_objects[i]
        # n_loc, n_objects, n_filters
        _pred_masked_sample = pred_masked[i, :, : _n_objects_sample]
        # _rand_masked_sample = rand_masked[i, :, : _n_objects_sample]
        # n_loc, n_objects, 1
        _gt_expanded_sample = gt_expanded[i, :, : _n_objects_sample]
        # _rand_sample = _rand_masked_sample.sum(0)

        _mean_sample = _pred_masked_sample.sum(
            0) / _gt_expanded_sample.sum(0)  # n_objects, n_filters
        if M=='intri':
            _mean_sample = _mean_sample / torch.norm(_mean_sample, 2, 1, True)
            # _rand_sample = _rand_sample / torch.norm(_rand_sample, 2, 1, True)
        if (max_n_objects - _n_objects_sample) != 0:
            n_fill_objects = int(max_n_objects - _n_objects_sample)
            _fill_sample = torch.zeros(n_fill_objects, n_filters)
            if usegpu:
                _fill_sample = _fill_sample.cuda()
            _fill_sample = Variable(_fill_sample)
            _mean_sample = torch.cat((_mean_sample, _fill_sample), dim=0)

        means.append(_mean_sample)
        #   norms.append()
        # rands.append(_rand_sample)

    means = torch.stack(means)

    # means = pred_masked.sum(1) / gt_expanded.sum(1)
    # # bs, n_instances, n_filters
    return means# , rands


def calculate_variance_term(pred, gt, means, n_objects, delta_v, norm=2, M='_intri'):
    """pred: bs, height * width, n_filters
       gt: bs, height * width, n_instances
       means: bs, n_instances, n_filters"""

    bs, n_loc, n_filters = pred.size()
    n_instances = gt.size(2)

    # bs, n_loc, n_instances, n_filters
    means = means.unsqueeze(1).expand(bs, n_loc, n_instances, n_filters)
    # bs, n_loc, n_instances, n_filters
    pred = pred.unsqueeze(2).expand(bs, n_loc, n_instances, n_filters)
    # bs, n_loc, n_instances, n_filters
    gt = gt.unsqueeze(3).expand(bs, n_loc, n_instances, n_filters)
    if M=='intri':
        with torch.no_grad():
            gt_sum = torch.sum(gt)
        var_term = ( gt_sum - torch.sum(means * pred * gt) ) / 2
    else:
        _var = (torch.clamp(torch.norm((pred - means), norm, 3) -
                        delta_v, min=0.0) ** 2) * gt[:, :, :, 0]

        var_term = 0.0
        for i in range(bs):
            _var_sample = _var[i, :, :n_objects[i]]  # n_loc, n_objects
            _gt_sample = gt[i, :, :n_objects[i], 0]  # n_loc, n_objects

            var_term += torch.sum(_var_sample) / torch.sum(_gt_sample)
    var_term = var_term / bs

    return var_term


def calculate_distance_term(means, n_objects, delta_d, norm=2, usegpu=True):
    """means: bs, n_instances, n_filters"""

    bs, n_instances, n_filters = means.size()

    dist_term = 0.0
    for i in range(bs):
        _n_objects_sample = int(n_objects[i])

        if _n_objects_sample <= 1:
            continue

        _mean_sample = means[i, : _n_objects_sample, :]  # n_objects, n_filters
        means_1 = _mean_sample.unsqueeze(1).expand(
            _n_objects_sample, _n_objects_sample, n_filters)
        means_2 = means_1.permute(1, 0, 2)

        diff = means_1 - means_2  # n_objects, n_objects, n_filters

        _norm = torch.norm(diff, norm, 2)

        margin = 2 * delta_d * (1.0 - torch.eye(_n_objects_sample))
        if usegpu:
            margin = margin.cuda()
        margin = Variable(margin)

        _dist_term_sample = torch.sum(
            torch.clamp(margin - _norm, min=0.0) ** 2)
        _dist_term_sample = _dist_term_sample / \
            (_n_objects_sample * (_n_objects_sample - 1))
        dist_term += _dist_term_sample

    dist_term = dist_term / bs

    return dist_term


def calculate_regularization_term(means, n_objects, norm):
    """means: bs, n_instances, n_filters"""

    bs, n_instances, n_filters = means.size()

    reg_term = 0.0
    for i in range(bs):
        _mean_sample = means[i, : n_objects[i], :]  # n_objects, n_filters
        _norm = torch.norm(_mean_sample, norm, 1)
        reg_term += torch.mean(_norm)
    reg_term = reg_term / bs

    return reg_term

def calculate_q_regularization_term(input, target):
    """input: bs, f, map"""
    target = torch.sum(target, dim=2, keepdim=True)
    with torch.no_grad():
        num = torch.sum(target)
        num = int(num)
    it = input * target
    #   l2 = torch.sum(torch.pow(it, 2), dim=2)
    l2 = torch.norm(it, 2, 2)
    # l = torch.pow(torch.sum(torch.pow(it, 2), dim = 2), 0.5)
    loss = torch.sum(torch.pow(l2 - 1, 2))/num
    return loss

def discriminative_loss(input, target, n_objects,
                        max_n_objects, delta_v, delta_d, norm, usegpu):
    """input: bs, n_filters, fmap, fmap
       target: bs, n_instances, fmap, fmap
       n_objects: bs"""
    #delta_v = 0.5 delta_d = 1.5 norm = 2
    alpha = beta = 1.0
    gamma = 0.005

    bs, n_filters, height, width = input.size()
    n_instances = target.size(1)

    input = input.permute(0, 2, 3, 1).contiguous().view(
        bs, height * width, n_filters)
    target = target.permute(0, 2, 3, 1).contiguous().view(
        bs, height * width, n_instances)

    cluster_means = calculate_means(
        input, target, n_objects, max_n_objects, usegpu)

    var_term = calculate_variance_term(
        input, target, cluster_means, n_objects, delta_v, norm)
    #   reg_term = calculate_regularization_term(cluster_means, n_objects, norm)
    reg_term = calculate_q_regularization_term(input, target)
    loss = alpha * var_term + gamma * reg_term

    return loss, cluster_means


class DiscriminativeLoss(_Loss):

    def __init__(self, delta_var, delta_dist, norm,
                 size_average=True, reduce=True, usegpu=True):
        pass
        super(DiscriminativeLoss, self).__init__(size_average)
        self.reduce = reduce

        # assert self.size_average
        # assert self.reduce

        self.delta_var = float(delta_var)
        self.delta_dist = float(delta_dist)
        self.norm = int(norm)
        self.usegpu = usegpu

        assert self.norm in [1, 2]

    def forward(self, input, target, n_objects, max_n_objects):
        # _assert_no_grad(target)
        return discriminative_loss(input, target, n_objects, max_n_objects,
                                   self.delta_var, self.delta_dist, self.norm,
                                   self.usegpu)
