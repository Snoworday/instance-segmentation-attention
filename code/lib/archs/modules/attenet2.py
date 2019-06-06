import torch.nn as nn
import torch
from modules.utils import SpatialAttentionLayer, ChannelAttentionLayer, HardAttentionLayer, L0Layer, Embedding, UpAttenLayer, UpDecoderLayer, select
# from modules.renet import ReNet
from losses import DiceLoss
from losses import lovasz_losses as L
from losses.multi_loss import FocalLoss
from losses.dice import decoder_mmd_loss
import config
from utils import onehot2idx, writeProJpg, writePnJpg
from torch.nn import Parameter
from torch.nn import BatchNorm2d as bn
import random
import cv2
import numpy as np

class DecoderLayer(nn.Module):
    def __init__(self):
        super(DecoderLayer, self).__init__()
        if False:
            self.ch1 = ChannelAttentionLayer(config.d_model, config.d_h)
            self.sp1 = SpatialAttentionLayer(config.d_model, config.d_h)
            self.ch2 = ChannelAttentionLayer(config.d_model, config.d_h)
            self.sp2 = SpatialAttentionLayer(config.d_model, config.d_h)
            self.pred = L0Layer(config.d_model, config.d_h)
        else:
            self.pred = L0Layer(64)
            self.bone = AttenDecoder()
        self.s_sp = SpatialAttentionLayer(config.d_model, config.d_h)
        self.attend = HardAttentionLayer(config.d_model, config.d_k, config.d_h)

        self.embedding = Embedding(config.d_model)
        self.criterion_focal = FocalLoss(gamma=config.FocalLoss_gamma)
        self.predloss = torch.nn.MSELoss(reduction='none')
        self.pnloss1 = nn.BCELoss(reduction='none')
        self.pnloss2 = FocalLoss(gamma=config.FocalLoss_gamma)
        self.criterion_ce = torch.nn.CrossEntropyLoss()
        self.evaluate_ce = torch.nn.CrossEntropyLoss()
        self.evaluate_dice = DiceLoss(
            optimize_bg=False, smooth=1.0, reduce=False).cuda()

        # self.criterion_ce = torch.nn.CrossEntropyLoss(reduction='none').cuda()
        self.criterion_dice = DiceLoss(
            optimize_bg=False, smooth=1.0, reduce=False).cuda()
        self.criterion_lovasz = L.StableBCELoss(reduction=False)

        self.baseline = 0
        self.lambda_r = config.lambda_r
        self.iter = 0
        self._fixed = None


    def init_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight.data)

            elif isinstance(m, bn):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
            elif isinstance(m, Parameter):
                m.data.fill_(0)

    @property
    def Parameters(self):
        return self.parameters

    def AlphaEntropy(self, alpha, mask=None):
        if len(alpha.size())==4:
            b, _ , h, w = alpha.shape
        else:
            b, h = alpha.shape[0], 256
        normal = (config.H / h) ** 0.5
        alpha = alpha.clamp(1e-7, 1. - 1e7)
        H = - alpha * torch.log(alpha) * normal
        if mask is None:
            mask = 1
        else:
            mask = mask.view(b, -1)
        H = torch.sum(H * mask)
        return H

    def Maskloss(self, pred, target, mask=None, loss_type='Multi', alpha=0, map_weight=0):
        b = pred.shape[0]
        if loss_type == 'Multi':
            target2 = torch.cat([1 - target, target], dim=1)

            dice_loss = self.criterion_dice(pred, target2, time=1, map_weight=map_weight)

            target = target.view(-1).long()
            pred = pred.permute(0,2,3,1).contiguous().view(-1, 2)
            if type(map_weight)!=int:
                map_weight = map_weight.view(-1)
            ce_loss = self.criterion_focal(pred, target, alpha, map_weight=map_weight)
            if mask is not None:
                ce_loss = ce_loss * mask.view(-1)
            ce_loss = torch.mean(ce_loss.view(b, -1), dim=1)
            loss_1 = config.CEWeight * ce_loss + dice_loss  # + mmd_loss
        # elif loss_type == 'lov':
        #     pred_ = torch.softmax(pred, dim=1)
        #     pred = pred_[:,1,:,:].contiguous().view(-1)
        #     target = target.view(-1)
        #     lov_loss = self.criterion_lovasz(pred, target)
        #     lov_loss = lov_loss * mask.view(-1)
        #     loss_1 = torch.mean(lov_loss.view(b, -1), dim=1) * config.LOVWeight
        return loss_1, dice_loss

    def set_fixed(self, input):
        _, h, w = input.shape
        idx_h = torch.linspace(0, w-1, w).unsqueeze(0).repeat(h, 1).cuda()
        idx_v = idx_h.t()
        self.fixed = [idx_h, idx_v]

    @property
    def fixed(self):
        if self._fixed is None:
            self._fixed = []
            H, W = config.H, config.W
            for i in range(5):
                h, w = H // (2**(4-i)), W // (2**(4-i))
                idx_h = torch.linspace(0, w - 1, w).unsqueeze(0).repeat(h, 1).detach().cuda()
                idx_v = idx_h.t()
                self._fixed.append([idx_h, idx_v])

        return self._fixed

    def Predloss(self, pred, mask=None, map_weight=0):
        loss = 0
        for p, t, f, w in zip(*(pred, mask, self.fixed, config.lambda_pyramid_weight)):
            if type(map_weight) != int:
                scale = map_weight.shape[2]//p.shape[2]
                map_weight_r = nn.MaxPool2d(kernel_size=scale, stride=scale)(map_weight)
            else:
                map_weight_r = map_weight
            multiloss, dice_loss = self.Maskloss(p, t, alpha=0.0, map_weight=map_weight_r)
            loss += multiloss * w # +  mmdloss * 0.03
            # loss += self.AlphaEntropy(p) * 10000
        return loss, dice_loss

    def vis(self, pred, target, pro, mask, alpha, alpha_sample, flist):
        for idx, f in enumerate(flist):
            re = 16 * 2 ** f
            pred__ = pred[f][:, 1] > pred[f][:, 0]
            m = (pred__[1] > 0.5).float()
            p = m.view(re, re, 1).cpu().numpy()
            m = np.concatenate([p, p, p], axis=2).astype(np.uint8) * 255
            # m[alpha_sample[1] // 256, alpha_sample[1] % 256, 2] = 0
            cv2.imwrite('p_%s.jpg' % f, m)

            re = 16 * 2 ** f
            predf = torch.softmax(pred[f], dim=1)
            region = predf[1][1].view(re, re)  # * (mask[1].view(256,256).float())
            Min, Max = region.min(), region.max()
            region[region != Max].max()
            region = (region - Min) / (Max - Min)
            m = region.view(re, re, 1).cpu().detach().numpy()
            m = (np.concatenate([m, m, m], axis=2) * 255).astype(np.uint8)
            # m[alpha_sample[1]//256, alpha_sample[1]%256] = 100
            cv2.imwrite('pred_%s.jpg' % f, m)
            region = target[f][1].view(re, re)  # * (mask[1].view(256,256).float())
            Min, Max = region.min(), region.max()
            region[region != Max].max()
            region = (region - Min) / (Max - Min)
            m = region.view(re, re, 1).cpu().detach().numpy()
            m = (np.concatenate([m, m, m], axis=2) * 255).astype(np.uint8)
            # m[alpha_sample[1]//256, alpha_sample[1]%256] = 100
            cv2.imwrite('target_%s.jpg' % f, m)

        writeProJpg(alpha, mask, 'pro.jpg', alpha_sample)
        writeProJpg(pro, mask, 'proall.jpg')
        writePnJpg(pro>0, mask)

        m = mask[1]
        t = m.view(256, 256, 1).cpu().numpy()
        m = np.concatenate([t, t, t], axis=2).astype(np.uint8) * 255
        cv2.imwrite('mas.jpg', m)


    def PNloss(self, *X):
        pred, advance, alpha, evaline, gold = X
        b, *_ = alpha.shape
        gold = gold.view(b, -1)
        if True:
            #   pnloss1
            alpha = alpha.view(b, -1)
            p = pred * alpha
            p = torch.softmax(p, dim=1)#p / torch.sum(p, dim=1, keepdim=True)
            p = torch.log(p.clamp(1e-7, 1. - 1e-7))
            pnloss1 = - p * advance
            #   pnloss2
            t = (alpha > evaline).float().view(b, -1)
            alpha= torch.sum(t, dim=1) / torch.sum(gold, dim=1)
            pred = pred.clamp(1e-7, 1. - 1e-7)
            F_loss_1 = -1 * (2-alpha).unsqueeze(1) * (1-pred.detach()) ** config.FocalLoss_gamma * torch.log(pred) * t * gold
            F_loss_0 = -1 * (alpha).unsqueeze(1) * pred.detach()**config.FocalLoss_gamma*torch.log(1-pred) * (1 - t) * gold
            pnloss2 = F_loss_1 + F_loss_0

            pnloss = torch.sum(pnloss1+0*pnloss2*0.3, dim=1) / b
        return pnloss

    def evaluate(self, pred, target, time=1):
        with torch.no_grad():
            pred = pred[-1]
            target = target[-1]
            ce_loss = self.evaluate_ce(pred.permute(0,2,3,1).contiguous().view(-1, 2), target.view(-1).long())
            target2 = torch.cat([1 - target, target], dim=1)
            dice_loss = self.evaluate_dice(pred, target2, time=time)
        return ce_loss, dice_loss

    def PNloss2(self, *X):
        pred, target, p_n, p_re, gold = X
        b = pred.shape[0]
        ploss = self.pnloss(pred, target)
        p_n = p_n * gold
        N = torch.sum(gold.view(b, -1), dim=1)
        with torch.no_grad():
            sel = (p_re < (1 / N[:, np.newaxis])).float()
            sel = sel.view(gold.shape) * gold
        nloss = - torch.log(1 - p_n + 1e-7) * sel
        nloss = torch.sum(nloss.view(b, -1), dim=1)
        pnloss = ploss*1.1 + nloss / torch.sum(sel.view(b, -1), dim=1)
        return pnloss

    def PNloss3(self, alpha_maxidx, pro, alpha, evaline, gold):
        b = pro.shape[0]
        p = torch.sum((pro * alpha_maxidx).view(b, -1), dim=1)
        ploss = torch.relu(-p)

        npoint = (alpha<evaline[:, None, None, None]).float()*gold
        N = torch.sum(npoint.view(b, -1), dim=1)
        n = pro*npoint
        nloss = torch.sum(torch.relu(n).view(b, -1), dim=1)
        pnloss = ploss + nloss/N
        return pnloss

    def Attenloss(self, pred, target, mask, pro, alpha, alpha_sample, alpha_maxidx, ratio, evaline, training, loss_type='Multi', use_mask=True):

        eval_ce, eval_dice = self.evaluate(pred, target)

        if not training:
            _, loss = self.evaluate(pred, target, time=2)
            criterion = eval_ce + eval_dice
            return loss, criterion, eval_ce, eval_dice
        if self.iter%40 == 0:
            try:
                self.vis(pred, target, pro, mask, alpha, alpha_sample, [0,1,2,3,4])
            except:
                pass
        self.iter += 1
        b = alpha.shape[0]

        loss_pred, dice_loss = self.Predloss(pred, target)
        target = target[-1]

        # print((pred[-1][:, 1] > pred[-1][:, 0]).sum())

        # pn_loss = self.PNloss3(alpha_maxidx, pro, alpha, evaline, target)

        #update baseline
        with torch.no_grad():
            ce_loss = self.criterion_ce(pred[-1].permute(0,2,3,1).contiguous().view(-1, 2), target.view(-1).long())
            log_p_y = - eval_dice.detach()
            self.baseline = 0.9 * self.baseline + 0.1*torch.mean(log_p_y)
        alpha = alpha.view(b, -1)
        log_p_s_a = torch.cat([alpha[i][alpha_sample[i]].unsqueeze(0) for i in range(b)])
        loss_2 = -(log_p_y - self.baseline) * torch.log(log_p_s_a)
        criterion = ce_loss + dice_loss.sum().detach()

        #shang
        if config.use_pyramid:
            H = self.AlphaEntropy(alpha, target.view(b, -1))
        else:
            H = self.AlphaEntropy(alpha, target.view(b, -1))

        # pn loss
        # roi_pred = p_n.view(b, -1)
        # roi_adv = (alpha.detach() - evaline.unsqueeze(1)).float() * target.view(b, -1)
        # print( 'p_point:', ((alpha.detach() > evaline.unsqueeze(1)).float()*target.view(b,-1)).sum(), 'n_point:', ((alpha.detach() < evaline.unsqueeze(1)).float()*target.view(b,-1)).sum())
        # pn_loss = self.PNloss(roi_pred, roi_adv.detach(), alpha.detach(), evaline.unsqueeze(1), target)

        # if isRandom:
        #     loss_pred = loss_pred.detach()
        loss = ratio * (config.lambda_l * loss_pred + config.lambda_r * loss_2) #+ config.lambda_pn * pn_loss # +  * loss_3

        loss = config.lambda_ins * (torch.sum(loss) - config.lambda_e * H) / b
        print('loss1:', config.lambda_l*loss_pred.data, 'loss2:', config.lambda_r*loss_2.data, 'H:', config.lambda_e*H.data, 'loss:', loss.data, 'criterion:', criterion.data)#,'pnloss',config.lambda_pn*pn_loss.data)
        return loss, criterion, eval_ce, eval_dice

    def select(self, map, point):
        if type(point)==list:
            b, c, h, w = map.shape
            map = map.permute(0,2,3,1).view(b, h*w, c)
            ans = torch.cat([map[i][point[i]] for i in range(b)])
            if len(ans.shape)==1:
                ans = ans.view(b, -1)

        else:
            ans = torch.sum((map*point), dim=[2, 3])
        return ans

    def sample(self, alpha_t, isRandom, alpha_g=None, training=True):
        b, _, h, w = alpha_t.shape
        alpha_t = alpha_t.cpu().view(b, -1)
        isgold = torch.sum(alpha_t, dim=1, keepdim=True)
        if isRandom:
            alpha_gold = alpha_g.cpu().view(b, -1)
            alpha_gold = alpha_gold * (alpha_t!=0).float() * isgold + alpha_gold * (1-isgold)
            alpha_gold = alpha_gold / torch.sum(alpha_gold, dim=1, keepdim=True)
            alpha_s = alpha_t + alpha_gold * 0.05  #   alpha_torch.pow(alpha_t, 2)
            alpha_s = alpha_s / torch.sum(alpha_s, dim=1, keepdim=True)
            s_t = torch.multinomial(alpha_s, 1)

        else:
            alpha_s = alpha_t
            if training:
                s_t = torch.multinomial(alpha_s, 1)
            else:
                s_t = torch.argmax(alpha_s, dim=1)

        s_t = list(map(int, list(s_t)))
        points = (list(map(lambda x:x//w, s_t)), list(map(lambda x:x%w, s_t)))

        if isRandom:
            s_t_ = [[i] for i in s_t]
            ratio = torch.gather(alpha_t, 1, torch.LongTensor(s_t_)) / torch.gather(alpha_s, 1, torch.LongTensor(s_t_))
            ratio = ratio.squeeze(1).cuda()
        else:
            ratio = 1
        return s_t, points, ratio

    def getDistribution(self, target, pred, idxs):
        '''

        :param target: b, n, h, w
        :param point:  int
        :return:
        '''
        B, _, h, w = target.shape
        t = torch.cat( [target[b, idxs[b], :, :].unsqueeze(0).float() for b in range(B)]).unsqueeze(1)
        p = torch.cat( [pred[b, idxs[b], :, :].unsqueeze(0) for b in range(B)]).unsqueeze(1)
        Evaline = 1 / torch.sum(t.view(B, -1), dim=1)
        _, alpha_maxidx = torch.max(p.view(B, -1), dim=1)
        alpha_maxidx = torch.zeros(B, h*w).cuda().scatter_(1, alpha_maxidx.unsqueeze(1), 1).reshape(B, 1, h, w)
        return t, p, Evaline.detach(), alpha_maxidx

    def getRandomIdx(self, n_ins):
        selected_idx= []
        for num in n_ins:
            l = list(range(num))
            random.shuffle(l)
            selected_idx.append(l)
        return selected_idx

    def forward(self, encode, mask, target, n_ins, training, X):
        '''

        :param encode: b, d_model, h, w
        :param mask: b,h,w
        :param target: b, n, h, w
        :return: loss
        '''
        b, c, h, w = encode.shape
        isRandom = False#training and random.random()< 1

        loss = 0
        iter = 0
        criterion = 0
        ins_ce_loss = 0
        ins_dice_loss = 0

        s_sp_out = self.s_sp(encode, mask)
        pro_split, pro_merge = self.attend(s_sp_out, mask, target)
        # maxIter = config.max_iter if training else min(n_ins).float()
        if training:
            maxIter = min(config.max_iter, int(min(n_ins)))
        else:
            maxIter = int(min(n_ins))
        selectedidx = self.getRandomIdx(n_ins)


        while iter < maxIter:
            idx = [i[iter] for i in selectedidx]

            gold, alpha, evaline, alpha_maxidx = self.getDistribution(target, pro_split, idx)
            with torch.no_grad():
                alpha_sample, alpha_sample_xy, ratio = self.sample(alpha, isRandom, gold, training)

            y_, pred_ = self.bone(alpha_sample_xy, X, mask, gold, pro_merge.detach(), training)

            loss_now, criterion_now, ce_loss, dice_loss = self.Attenloss(pred_, y_, mask, pro_merge, alpha, alpha_sample, alpha_maxidx, ratio, evaline, training)
            loss += loss_now
            criterion += criterion_now
            ins_ce_loss += ce_loss
            ins_dice_loss += torch.mean(dice_loss)
            # print('iter:', iter)
            iter += 1
        loss = torch.mean(loss / maxIter)
        criterion = torch.mean(criterion / maxIter)
        ins_ce_loss = ins_ce_loss / maxIter
        ins_dice_loss = ins_dice_loss / maxIter


        print(loss.data, criterion.data, ins_ce_loss.data, ins_dice_loss.data)
        return loss, criterion, ins_ce_loss, ins_dice_loss


class AttenDecoder(nn.Module):
    def __init__(self):
        super(AttenDecoder, self).__init__()
        if config.use_encode:
            self.upAtten0 = UpDecoderLayer([512], 256, loop=2, factor=16, is_first=True, use_mask=True)
            self.upAtten1 = UpDecoderLayer([256, 256], 128, loop=2, factor=8, use_mask=True)
            self.upAtten2 = UpDecoderLayer([128, 128], 64, loop=2, factor=4, use_mask=True)
            self.upAtten3 = UpDecoderLayer([64, 64], 32, loop=2, factor=2, use_mask=True)
            self.upAtten4 = UpDecoderLayer([32, 32], 32, loop=2, factor=1, is_last=True, use_mask=True)
        else:
            self.upAtten0 = UpDecoderLayer(256, 128, loop=2, factor=16, is_first=True)
            self.upAtten1 = UpDecoderLayer(256, 64, loop=2, factor=8)
            self.upAtten2 = UpDecoderLayer(128, 32, loop=2, factor=4)
            self.upAtten3 = UpDecoderLayer(64, 32, loop=2, factor=2)
            self.upAtten4 = UpDecoderLayer(64, 32, loop=2, factor=1)
        # self.outc = nn.Conv2d(64, 2, 1)
    def concat(self, pred_pre, pred_now, pred_all):
        b, _, h, w = pred_now.shape
        # out = torch.cat([nn.functional.interpolate(f, (h, w), mode='bilinear', align_corners=False) for f in input], dim=1)
        # pred_all = [nn.functional.interpolate(f, (h, w), mode='bilinear', align_corners=False) for f in pred_all] + [pred_now]
        pred_all.append(pred_now)
        pred_out = pred_now
        # pred_out = nn.functional.interpolate(pred_pre, (h, w), mode='bilinear', align_corners=False) * (1-config.decay) + pred_now * config.decay
        return pred_all, pred_out

    def relocal(self, points, h, w):
        try:
            for i in range(len(points)):
                    points[0][i], points[1][i] = points[0][i]*config.H//h, points[1][i]*config.W//w
        except:
            a=1
        return points

    def forward(self, alpha_sample, x, mask, gold=None, anchorpro=None, training=True):
        '''

        :param alpha_sample: 采样点(256*256)
        :param x: 5个尺寸的特征图
        :param mask: ins_seg(256*256)
        :return:
        '''
        x1, x2, x3, x4, x5 = x
        pred_all = []
        target_all = []
        h, w = x1.shape[2], x1.shape[3]

        alpha_sample = self.relocal(alpha_sample, h, w)
        if config.use_pyramid:
            x, pred_0, target_0 = self.upAtten0(None, x5, alpha_sample, [gold, anchorpro, mask], mask, selection=False, training=training)
            pred_all.append(pred_0)

            x, pred_1, target_1 = self.upAtten1(x, x4, alpha_sample, [gold, anchorpro, mask], [target_0, pred_0], training=training)
            pred_all, pred_01 = self.concat(pred_0, pred_1, pred_all)

            x, pred_2, target_2 = self.upAtten2(x, x3, alpha_sample, [gold, anchorpro, mask], [target_1, pred_01], training=training)
            pred_all, pred_12 = self.concat(pred_1, pred_2, pred_all)

            x, pred_3, target_3 = self.upAtten3(x, x2, alpha_sample, [gold, anchorpro, mask], [target_2, pred_12], training=training)
            pred_all, pred_23 = self.concat(pred_2, pred_3, pred_all)

            x, pred_4, target_4 = self.upAtten4(x, x1, alpha_sample, [gold, anchorpro, mask], [target_3, pred_23], training=training)
            pred_all, _ = self.concat(pred_3, pred_4, pred_all)
            target_all = [target_0, target_1, target_2, target_3, target_4]
            return target_all, pred_all
            # return (target_0, target_1, target_2, target_3, target_4), (pred_0, pred_1, pred_2, pred_3, pred_4)
        else:
            x_1 = self.upAtten0(None, x5, alpha_sample, mask)
            x_2 = self.upAtten1(x_1, x4, alpha_sample, mask)

            x_12 = self.concat(x_1, x_2)
            x_3 = self.upAtten2(x_12, x3, alpha_sample, mask)

            x_23 = self.concat(x_2, x_3)
            x_4 = self.upAtten3(x_23, x2, alpha_sample, mask)

            x_34 = self.concat(x_3, x_4)
            x_5 = self.upAtten4(x_34, x1, alpha_sample, mask)

            x_concat = self.concat(x_4, x_5)
            pred = self.pred(x_concat, mask)
            target = gold
            return target, pred
        # x = self.outc(x)

