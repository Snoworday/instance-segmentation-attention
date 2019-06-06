import sys
DEBUG = False
sys.path = sys.path + ['/media/snowday/045A0A095A09F7E6/git/segmentation/instance-segmentation/code', '/media/snowday/045A0A095A09F7E6/git/segmentation/instance-segmentation/code/lib',\
  '/media/snowday/045A0A095A09F7E6/git/segmentation/instance-segmentation/code/lib/archs','/media/snowday/045A0A095A09F7E6/git/segmentation/instance-segmentation/code/lib/losses',\
  '/media/snowday/045A0A095A09F7E6/git/segmentation/instance-segmentation/code/lib/archs/modules',\
  '/media/snowday/045A0A095A09F7E6/git/segmentation/instance-segmentation/code/settings/CVPPP']
import argparse
import random
import os
import getpass
import datetime
import shutil
import numpy as np
import torch
from torchvision import datasets, transforms
from lib import SegDataset, TransferDataset, Model, AlignCollate

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='/media/snowday/045A0A095A09F7E6/git/segmentation/instance-segmentation/models/CVPPP/2018-12-2_23-32_snowday_21-499754/model_374_0.09648385643959045.pth',
                    help="Filepath of trained model (to continue training) \
                         [Default: '']")
parser.add_argument('--usegpu', action='store_true', default = True,
                    help='Enables cuda to train on gpu [Default: False]')
parser.add_argument('--nepochs', type=int, default=800,
                    help='Number of epochs to train for [Default: 600]')
parser.add_argument('--batchsize', type=int,
                    default=2, help='Batch size [Default: 2]')
parser.add_argument('--debug', action='store_true',
                    help='Activates debug mode [Default: False]')
parser.add_argument('--nworkers', type=int,
                    help='Number of workers for data loading \
                        (0 to do it using main process) [Default : 2]',
                    default=2)
parser.add_argument('--dataset', type=str, default='CVPPP',
                    help='Name of the pythondataset which is "CVPPP"',
                    required=False)
opt = parser.parse_args()

assert opt.dataset in ['CVPPP', ]

if opt.dataset == 'CVPPP':
    from settings import CVPPPTrainingSettings
    ts = CVPPPTrainingSettings()


def generate_run_id():

    username = getpass.getuser()

    now = datetime.datetime.now()
    date = map(str, [now.year, now.month, now.day])
    coarse_time = map(str, [now.hour, now.minute])
    fine_time = map(str, [now.second, now.microsecond])

    run_id = '_'.join(['-'.join(date), '-'.join(coarse_time),
                       username, '-'.join(fine_time)])
    return run_id


RUN_ID = generate_run_id()
model_save_path = os.path.abspath(os.path.join(os.path.abspath(__file__),
                                               os.path.pardir, os.path.pardir,
                                               'models', opt.dataset, RUN_ID))
os.makedirs(model_save_path)

CODE_BASE_DIR = os.path.abspath(os.path.join(
    os.path.abspath(__file__), os.path.pardir))
shutil.copytree(os.path.join(CODE_BASE_DIR, 'settings'),
                os.path.join(model_save_path, 'settings'))
shutil.copytree(os.path.join(CODE_BASE_DIR, 'lib'),
                os.path.join(model_save_path, 'lib'))

if torch.cuda.is_available() and not opt.usegpu:
    print( 'WARNING: You have a CUDA device, so you should probably \
        run with --usegpu')

# Load Seeds
random.seed(ts.SEED)
np.random.seed(ts.SEED)
torch.manual_seed(ts.SEED)

# Define Data Loaders
pin_memory = False
if opt.usegpu:
    pin_memory = True

train_dataset = SegDataset(ts.TRAINING_LMDB)
assert train_dataset

train_align_collate = AlignCollate(
    'training',
    ts.N_CLASSES,
    ts.MAX_N_OBJECTS,
    ts.MEAN,
    ts.STD,
    ts.IMAGE_HEIGHT,
    ts.IMAGE_WIDTH, ts.wae_opt[0]['out_shape'],
    random_hor_flipping=ts.HORIZONTAL_FLIPPING,
    random_ver_flipping=ts.VERTICAL_FLIPPING,
    random_transposing=ts.TRANSPOSING,
    random_90x_rotation=ts.ROTATION_90X,
    random_rotation=ts.ROTATION,
    random_color_jittering=ts.COLOR_JITTERING,
    random_grayscaling=ts.GRAYSCALING,
    random_channel_swapping=ts.CHANNEL_SWAPPING,
    random_gamma=ts.GAMMA_ADJUSTMENT,
    random_resolution=ts.RESOLUTION_DEGRADING,
    center_cut = ts.CENTER_CUT
)

train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=opt.batchsize,
                                           shuffle=True,
                                           num_workers=opt.nworkers,
                                           pin_memory=pin_memory,
                                           collate_fn=train_align_collate)

test_dataset = SegDataset(ts.VALIDATION_LMDB)
assert test_dataset

test_align_collate = AlignCollate(
    'test',
    ts.N_CLASSES,
    ts.MAX_N_OBJECTS,
    ts.MEAN,
    ts.STD,
    ts.IMAGE_HEIGHT,
    ts.IMAGE_WIDTH, ts.wae_opt[0]['out_shape'],
    random_hor_flipping=ts.HORIZONTAL_FLIPPING,
    random_ver_flipping=ts.VERTICAL_FLIPPING,
    random_transposing=ts.TRANSPOSING,
    random_90x_rotation=ts.ROTATION_90X,
    random_rotation=ts.ROTATION,
    random_color_jittering=ts.COLOR_JITTERING,
    random_grayscaling=ts.GRAYSCALING,
    random_channel_swapping=ts.CHANNEL_SWAPPING,
    random_gamma=ts.GAMMA_ADJUSTMENT,
    random_resolution=ts.RESOLUTION_DEGRADING,
    center_cut = ts.CENTER_CUT
)

test_loader = torch.utils.data.DataLoader(test_dataset,
                                          batch_size=opt.batchsize,
                                          shuffle=False,
                                          num_workers=opt.nworkers,
                                          pin_memory=pin_memory,
                                          collate_fn=test_align_collate)
if DEBUG == True:
    transform = transforms.Compose(
        [transforms.Resize([256, 256]),
         transforms.RandomCrop(224),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor()])
    data1 = datasets.ImageFolder(root='/media/snowday/045A0A095A09F7E6/data/office31/amazon', transform=transform)
    tmp_align_collate = AlignCollate(
        'training',
        ts.N_CLASSES,
        ts.MAX_N_OBJECTS,
        ts.MEAN,
        ts.STD,
        ts.IMAGE_HEIGHT,
        ts.IMAGE_WIDTH,
        random_hor_flipping=ts.HORIZONTAL_FLIPPING,
        random_ver_flipping=ts.VERTICAL_FLIPPING,
        random_transposing=ts.TRANSPOSING,
        random_90x_rotation=ts.ROTATION_90X,
        random_rotation=ts.ROTATION,
        random_color_jittering=ts.COLOR_JITTERING,
        random_grayscaling=ts.GRAYSCALING,
        random_channel_swapping=ts.CHANNEL_SWAPPING,
        random_gamma=ts.GAMMA_ADJUSTMENT,
        random_resolution=ts.RESOLUTION_DEGRADING,
        center_cut = ts.CENTER_CUT
    )
    tmp_loader = torch.utils.data.DataLoader(data1,
                                               batch_size=opt.batchsize,
                                               shuffle=True,
                                               num_workers=opt.nworkers,
                                               pin_memory=pin_memory,
                                               collate_fn=tmp_align_collate)
    itr1 = iter(tmp_loader)
    a = itr1.next()
    train_iter = iter(train_loader)
    a = train_iter.next()

    transfer_dataset = TransferDataset(ts.trans_data_path)
    tmp = transfer_dataset[16]

# Define Model
model = Model(opt.dataset, ts.MODEL_NAME, ts.N_CLASSES, ts.MAX_N_OBJECTS, ts.wae_opt,
              use_instance_segmentation=ts.USE_INSTANCE_SEGMENTATION, use_wae = ts.USE_WAE,
              use_coords=ts.USE_COORDINATES, load_model_path=ts.load_model_path, load_decoder_model_path=ts.load_decoder_model_path,
              usegpu=opt.usegpu)

# Train Model
model.fit(ts.CRITERION, ts.DELTA_VAR, ts.DELTA_DIST, ts.NORM, ts.LEARNING_RATE,
          ts.WEIGHT_DECAY, ts.CLIP_GRAD_NORM, ts.LR_DROP_FACTOR,
          ts.LR_DROP_PATIENCE, ts.OPTIMIZE_BG, ts.OPTIMIZER, ts.TRAIN_CNN,
          opt.nepochs, ts.CLASS_WEIGHTS, train_loader, test_loader,
          model_save_path, opt.debug)
