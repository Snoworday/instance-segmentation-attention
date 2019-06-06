H, W = 256, 256
train = True
gamma = 0.99
batch_size = 2
pickle_path = '/media/snowday/045A0A095A09F7E6/git/segmentation/instance-segmentation-atten/code/'
#attn
atten_cost_weight = 1
max_iter = 32

#dqn
buffer_start = 20
buffer_capacity = 60
dqn_batch_size = 4

#decoder
FocalLoss_gamma = 2
# FocalLoss_alpha = 1
CEWeight = 10
LOVWeight = 10
decoer_num_layers = 1

n_head = 2
d_model = 24
d_k = 12
d_v = 12
d_inner = 40

epsilon_start, epsilon_end, epsilon_decay = 1, 0.01, 500

#nomask 101
#mask 010

# DenseASPP
d_h = 20
d_features0 = 20
d_features1 = 10
dropout0 = 0.1
dropout1 = 0.1

# mmd_loss
mmd_l = 64
mmd_pz = 'normal'
mmd_zdim = d_model

lambda_l = 0.5
lambda_r = 2
lambda_e = 5
lambda_pn = 0.01
lambda_ins = 1
lambda_pred_w = 1 # 0.0001
lambda_pyramid_weight = [16, 8, 4, 2, 1] # [0.3, 0.6, 1, 1.4, 1.7]          [1.7, 1.4, 1, 0.6, 0.3]

#img
shape = (256, 256)

max_iter = 2

use_mask = True
use_encode = True
use_pyramid = True

type = 1
nonLocalType = 'Concatenation'
drop_rate = 0.5
decay = 1
selection_threshold = 0.02

positionType = 1