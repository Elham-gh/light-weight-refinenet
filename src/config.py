import numpy as np

# DATASET PARAMETERS
TRAIN_DIR = '/content/drive/MyDrive/datasets/nyudv2/'
VAL_DIR = TRAIN_DIR
TRAIN_LIST = ['/content/light-weight-refinenet/data/train_CEN.nyu'] * 6
VAL_LIST = ['/content/light-weight-refinenet/data/val_CEN.nyu'] * 6
BPD_DIR = ['/content/drive/MyDrive/datasets/nyu/CEN_bpds.pkl'] * 6
SHORTER_SIDE = [350] * 6
CROP_SIZE = [500] * 6
NORMALISE_PARAMS = [
    1.0 / 255,  # SCALE
    np.array([0.485, 0.456, 0.406]).reshape((1, 1, 3)),  # MEAN
    np.array([0.229, 0.224, 0.225]).reshape((1, 1, 3)),
]  # STD
BATCH_SIZE = [6] * 6
BATCH_MEAN = 12
NUM_WORKERS = 4
NUM_CLASSES = [40] * 6
LOW_SCALE = [0.5] * 6
HIGH_SCALE = [2.0] * 6
IGNORE_LABEL = 255

# ENCODER PARAMETERS
ENC = "50"
ENC_PRETRAINED = True  # pre-trained on ImageNet or randomly initialised

# GENERAL
EVALUATE = False
FREEZE_BN = [True] * 1
NUM_SEGM_EPOCHS = [100] * 6
PRINT_EVERY = 10
RANDOM_SEED = 42
CKPT_PATH = '/content/drive/MyDrive/Super-BPD/LWR/ckpt_att/'
VAL_EVERY = [10] * 6  # how often to record validation scores
RESUME = '' #CKPT_PATH

# OPTIMISERS' PARAMETERS
LR_ENC = [1e-3, 5e-4, 2.5e-4, 1e-4]  # TO FREEZE, PUT 0
LR_DEC = [1e-2, 5e-3, 2.5e-3, 1e-3]

MOM_ENC = [0] * 6  # TO FREEZE, PUT 0
MOM_DEC = [0] * 6
WD_ENC = [1e-5] * 6  # TO FREEZE, PUT 0
WD_DEC = [1e-5] * 6
OPTIM_DEC = "sgd"
