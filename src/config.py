import numpy as np

# DATASET PARAMETERS
TRAIN_DIR = '/content/drive/MyDrive/datasets/nyudv2/'
VAL_DIR = TRAIN_DIR
TRAIN_LIST = ['/content/light-weight-refinenet/data/train_CEN.nyu'] * 1
VAL_LIST = ['/content/light-weight-refinenet/data/val_CEN.nyu'] * 1
BPD_DIR = ['/content/drive/MyDrive/datasets/nyu/CEN_bpds.pkl'] * 1
SHORTER_SIDE = [350] * 1
CROP_SIZE = [500] * 1
NORMALISE_PARAMS = [
    1.0 / 255,  # SCALE
    np.array([0.485, 0.456, 0.406]).reshape((1, 1, 3)),  # MEAN
    np.array([0.229, 0.224, 0.225]).reshape((1, 1, 3)),
]  # STD
BATCH_SIZE = [1] * 1
BATCH_MEAN = 12
NUM_WORKERS = 4
NUM_CLASSES = [40] * 1
LOW_SCALE = [0.5] * 1
HIGH_SCALE = [2.0] * 1
IGNORE_LABEL = 255

# ENCODER PARAMETERS
ENC = "50"
ENC_PRETRAINED = True  # pre-trained on ImageNet or randomly initialised

# GENERAL
EVALUATE = False
FREEZE_BN = [True] * 1
NUM_SEGM_EPOCHS = [1000] * 1
PRINT_EVERY = 10
RANDOM_SEED = 42
CKPT_PATH = '/content/drive/MyDrive/Super-BPD/LWR/ckpt_att_1C/'
VAL_EVERY = [10] * 1  # how often to record validation scores
RESUME = ''#'/content/drive/MyDrive/Super-BPD/LWR/ckpt_att_batch/'

# OPTIMISERS' PARAMETERS
LR_ENC = [1e-9]  # TO FREEZE, PUT 0
LR_DEC = [1e-8]

MOM_ENC = [0] * 1  # TO FREEZE, PUT 0
MOM_DEC = [0] * 1
WD_ENC = [1e-5] * 1  # TO FREEZE, PUT 0
WD_DEC = [1e-5] * 1
OPTIM_DEC = "sgd"
