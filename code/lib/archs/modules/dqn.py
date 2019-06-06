from torch import nn
import torch
import numpy as np
import math
import random
import torch.nn.functional as F
from torch import optim
from modules.utils import Encoder
import config
import os
from collections import deque
# https://discuss.pytorch.org/t/list-of-nn-module-in-a-nn-module/219


class RLSelect(nn.Module):
    '''duelingDQN'''
    def __init__(self, is_training, channel=config.d_model, num_bakeup=10):
        super(RLSelect, self).__init__()
        self.num_bakeup = num_bakeup
        self.batch = []
        self.frame_id = 0
        self.state_dim = channel

        self.layers_stack = nn.ModuleList([
            self.conv_dw(self.state_dim, 8),
            self.conv_dw(8, 12),
            self.conv_dw(12, 6)
        ])
        self.last = nn.Conv2d(6, 1, 1, 1)
        if is_training:
            for param in self.parameters():
                param.requires_grad = True
            self.train()
        else:
            for param in self.parameters():
                param.requires_grad = False
            self.eval()

    def conv_dw(self, inp, oup):
        return nn.Sequential(
            nn.Conv2d(inp, inp, kernel_size=3, stride=1, groups=inp, padding=1, bias=False),
            nn.BatchNorm2d(inp),
            nn.ReLU6(inplace=True),
            nn.Conv2d(inp, oup, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(oup),
            nn.ReLU6(inplace=True)
        )

    def forward(self, input, mask):
        '''input: b, 256*256, 512
           ins_seg: b, 256*256, 32
           mask: b, 256*256
            out: b, 256*256
            out_idx: b,
        '''
        # b, c, L = input.shape
        # l = int(L**0.5)
        feature = input
        b, c, h, w = input.shape
        mask = mask.view(b, 1, h, w)
        # input = input.contiguous()
        # input = input.view(b * l, c)

        for layer in self.layers_stack:
            feature = feature * mask
            feature = layer(feature)
        feature = self.last(feature)
        q_values = feature.view(b, h*w)
        return q_values

    def act(self, epsilon, state, state_idx, state_input, mask):
        batch = mask.shape[0]
        actions = []
        out = []
        out_idx = []

        state_idx = state_idx.permute(0, 2, 1)
        q_values = self.forward(state, mask)
        for b in range(batch):
            cand = mask[b].nonzero()
            if random.random() < epsilon:
                action = cand[int(random.random() * int(cand.shape[0]))]
            else:
                try:
                    action = cand[q_values[b][cand].argmax()]
                except:
                    return 1, 2, 3
            actions.append(action)
            out.append(state_input[b][action])
            out_idx.append(state_idx[b][action].argmax())
        actions = torch.cat(actions)
        out = torch.cat(out)
        self.frame_id += 1
        out_idx = torch.LongTensor(out_idx).cuda()
        return out, out_idx, actions

class ReplayBuffer(object):
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    # def push(self, state, action, reward, next_state, done):
    #     state = np.expand_dims(state, 0)
    #     next_state = np.expand_dims(next_state, 0)
    #
    #     self.buffer.append((state, action, reward, next_state, done))
    def push(self, input):  #   state_feature, action, reward, mask, next_mask, done):
        '''input_list, candidate_idx, iou, mask, mask_, done'''
        b = list(zip(*input))
        self.buffer += b
    def sample(self, batch_size):
        state, action, reward, mask, next_mask, done = zip(*random.sample(self.buffer, batch_size))
        return state, action, reward, mask, next_mask, done
        # return np.concatenate(state), action, reward, np.concatenate(next_state), done
    def __len__(self):
        return len(self.buffer)

class DQNSelecter(nn.Module):
    def __init__(self, attenet, load_path=None, usegpu=True):
        super(DQNSelecter, self).__init__()
        self.selecter = RLSelect(is_training=True)
        self.target = RLSelect(is_training=False)

        self.usegpu = usegpu
        if usegpu:
            self.selecter = self.selecter.cuda()
            self.target = self.target.cuda()
        self.optimizer = optim.Adam(self.selecter.parameters())
        self.buffer = ReplayBuffer(config.buffer_capacity)
        if load_path:
            self.load_model_path = load_path
            self.__load_weights()

    @property
    def frame(self):
        return self.selecter.frame_id

    @property
    def epsilon(self):
        return config.epsilon_end+(config.epsilon_start-config.epsilon_end)*math.exp(-1.*self.frame/config.epsilon_decay)

    def toTensor(self, *input):
        out = [torch.cat([i.unsqueeze(0) for i in item]) for item in input]
        return out

    def compute_loss(self, batch_size = config.dqn_batch_size):
        curM, tarM = self.selecter, self.target

        state, action, reward, mask, next_mask, done = self.buffer.sample(batch_size)
        state, action, reward, mask, next_mask, done = self.toTensor(state, action, reward, mask, next_mask, done)

        nomask = 1 - mask
        nonext_mask = 1 - next_mask

        q_values = curM(state, mask)
        q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
        next_q_values = curM(  state, next_mask)
        next_q_state_values = tarM( state, next_mask)

        next_q_value = next_q_state_values.gather(1, torch.max(next_q_values.masked_fill(nonext_mask.byte(), -np.inf), 1)[1].unsqueeze(1)).squeeze(1)

        expected_q_value = reward + config.gamma * next_q_value * (1-done.float())

        loss = (q_value - expected_q_value.data).pow(2).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def push(self, *input):
        self.buffer.push(input)

    def act(self, *input):
        return self.selecter.act(self.epsilon, *input)

    def update(self):
        if len(self.buffer) >= config.buffer_start:
            self.compute_loss()
        if self.frame % 100 == 0:
            self.update_target(self.selecter, self.target)

    def update_target(self, curM, tarM):
        tarM.load_state_dict(curM.state_dict())

    def save_weights(self, path):
        torch.save(self.target.state_dict(), path)

    def __load_weights(self):

        if self.load_model_path != '':
            assert os.path.isfile(self.load_model_path), 'Model : {} does not \
                exists!'.format(self.load_model_path)
            print( 'Loading model from {}'.format(self.load_model_path))

            model_state_dict = self.selecter.state_dict()

            if self.usegpu:
                pretrained_state_dict = torch.load(self.load_model_path)
            else:
                pretrained_state_dict = torch.load(
                    self.load_model_path, map_location=lambda storage,
                    loc: storage)

            model_state_dict.update(pretrained_state_dict)
            self.selecter.load_state_dict(model_state_dict)
            self.target.load_state_dict(model_state_dict)