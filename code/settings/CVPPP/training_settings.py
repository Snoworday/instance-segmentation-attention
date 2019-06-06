import os
from model_settings import ModelSettings


class TrainingSettings(ModelSettings):

    def __init__(self):
        super(TrainingSettings, self).__init__()

        self.TRAINING_LMDB = os.path.join(
            self.BASE_PATH,
            'data',
            'processed',
            'CVPPP',
            'lmdb',
            'training-lmdb')
        self.VALIDATION_LMDB = os.path.join(
            self.BASE_PATH,
            'data',
            'processed',
            'CVPPP',
            'lmdb',
            'validation-lmdb')
        self.trans_data_path = '/media/snowday/045A0A095A09F7E6/data/leafsnap-dataset/dataset/images'
        self.TRAIN_CNN = True

        self.OPTIMIZER = 'Adadelta'
        # optimizer - one of : 'RMSprop', 'Adam', 'Adadelta', 'SGD'
        self.LEARNING_RATE = 1
        self.LR_DROP_FACTOR = 0.5
        self.LR_DROP_PATIENCE = 25
        self.WEIGHT_DECAY = 0.001
        # weight decay - use 0 to disable it
        self.CLIP_GRAD_NORM = 10.0
        # max l2 norm of gradient of parameters - use 0 to disable it

        self.HORIZONTAL_FLIPPING = True
        self.VERTICAL_FLIPPING = True
        self.TRANSPOSING = True
        self.ROTATION_90X = True
        self.ROTATION = True
        self.COLOR_JITTERING = False
        self.GRAYSCALING = False
        self.CHANNEL_SWAPPING = False
        self.GAMMA_ADJUSTMENT = False
        self.RESOLUTION_DEGRADING = False

        self.CRITERION = 'Multi'
        # criterion - One of 'CE', 'Dice', 'Multi'
        self.OPTIMIZE_BG = False

        self.CENTER_CUT = True
        self.SEED = 23
        self.USE_WAE = False
        self.wae_opt = [{'coding': 20, 'num_layers': 4, 'out_shape': [64, 64, 1], 'num_units': 512, 'wae_weight': 10,\
                         'decoder_lr':{ 'optimizer':'Adam', 'learning_rate': 0.001, 'weight_decay': 0.001,'lr_drop_patience': 20, 'lr_drop_factor': 0.8}},
                        {'lambda': 1, 'pz': 'uniform', 'zdim': 20, 'num_point':24, 'pz_scale': 1, 'noise_num':64}]
        # coding = zdim
        self.load_model_path = ''#/media/snowday/045A0A095A09F7E6/git/segmentation/instance-segmentation-atten/models/CVPPP/2019-4-11_16-49_snowday_31-430351/model_35_0.278393417596817_1.pth'#/media/snowday/045A0A095A09F7E6/git/segmentation/instance-segmentation-atten/models/CVPPP/2019-4-11_16-49_snowday_31-430351/model_118_0.17253360152244568_1.pth'#/media/snowday/045A0A095A09F7E6/git/segmentation/instance-segmentation-atten/models/perfect/2019-4-4_4-36_snowday_36-90450/model_344_0.14633582532405853_0.03125.pth'#/media/snowday/045A0A095A09F7E6/git/segmentation/instance-segmentation-atten/models/perfect/2019-3-26_9-56_snowday_18-469276/model_194_0.14564888179302216_0.03125.pth'#/media/snowday/045A0A095A09F7E6/git/segmentation/instance-segmentation-atten/models/CVPPP/2019-4-11_1-31_snowday_57-107581/model_196_0.18067385256290436_0.0625.pth'#/media/snowday/045A0A095A09F7E6/git/segmentation/instance-segmentation-atten/models/CVPPP/2019-4-6_20-9_snowday_13-960582/model_54_0.2765701115131378_0.05.pth'#/media/snowday/045A0A095A09F7E6/git/segmentation/instance-segmentation-atten/models/CVPPP/2019-4-4_4-36_snowday_36-90450/model_344_0.14633582532405853_0.03125.pth'#/media/snowday/045A0A095A09F7E6/git/segmentation/instance-segmentation-atten/models/CVPPP/2019-4-5_3-9_snowday_50-688433/model_34_0.43103376030921936_1.pth'#/media/snowday/045A0A095A09F7E6/git/segmentation/instance-segmentation-atten/models/CVPPP/2019-4-4_4-36_snowday_36-90450/model_344_0.14633582532405853_0.03125.pth'#/media/snowday/045A0A095A09F7E6/git/segmentation/instance-segmentation-atten/models/CVPPP/2019-4-3_14-26_snowday_23-86646/model_9_0.5666579604148865_1.pth'#/media/snowday/045A0A095A09F7E6/git/segmentation/instance-segmentation-atten/models/perfect/2019-3-26_9-56_snowday_18-469276/model_194_0.14564888179302216_0.03125.pth'#/media/snowday/045A0A095A09F7E6/git/segmentation/instance-segmentation-atten/models/perfect/2019-3-26_9-56_snowday_18-469276'
        self.load_decoder_model_path = '' #''/media/snowday/045A0A095A09F7E6/git/segmentation/instance-segmentation-atten/models/CVPPP/2018-12-18_21-35_snowday_34-950746/dqn_model_101_0.919108510017395.pth'   # '/media/snowday/045A0A095A09F7E6/git/segmentation/instance-segmentation-atten/models/CVPPP/2018-12-18_13-30_snowday_58-361107/dqn_model_2_15.803201675415039.pth'
