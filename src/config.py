import numpy as np

# DATASET PARAMETERS
TRAIN_DIR = '/content/drive/MyDrive/datasets/nyudv2/'
VAL_DIR = TRAIN_DIR
TRAIN_LIST = ['/content/light-weight-refinenet/data/train_CEN.nyu'] * 12
VAL_LIST = ['/content/light-weight-refinenet/data/val_CEN.nyu'] * 12
BPD_DIR = ['/content/drive/MyDrive/datasets/nyu/CEN_bpds.pkl'] * 12
SHORTER_SIDE = [350] * 12
CROP_SIZE = [500] * 12
NORMALISE_PARAMS = [
    1.0 / 255,  # SCALE
    np.array([0.485, 0.456, 0.406]).reshape((1, 1, 3)),  # MEAN
    np.array([0.229, 0.224, 0.225]).reshape((1, 1, 3)),
]  # STD
BATCH_SIZE = [6] * 12
NUM_WORKERS = 16
NUM_CLASSES = [40] * 3
LOW_SCALE = [0.5] * 12
HIGH_SCALE = [2.0] * 12
IGNORE_LABEL = 255

# ENCODER PARAMETERS
ENC = "50"
ENC_PRETRAINED = True  # pre-trained on ImageNet or randomly initialised

# GENERAL
EVALUATE = False
FREEZE_BN = [True] * 12
NUM_SEGM_EPOCHS = [100] * 12
PRINT_EVERY = 10
RANDOM_SEED = 42
CKPT_PATH = "/content/drive/MyDrive/Super-BPD/LWR/ckpt_multimodalcat/"
RESUME = '' #CKPT_PATH + "353_19/Copy of "
VAL_EVERY = [10] * 12  # how often to record validation scores

# OPTIMISERS' PARAMETERS
LR_ENC = [5e-5, 2.5e-5, 2.5e-5, 1e-5, 1e-5, 5e-6, 5e-6, 2.5e-6, 2.5e-6, 1e-6, 1e-6, 5e-7]  # TO FREEZE, PUT 0
LR_DEC = [5e-4, 2.5e-4, 2.5e-4, 1e-4, 1e-4, 5e-5, 5e-5, 2.5e-5, 2.5e-5, 1e-5, 1e-5, 5e-6]
MOM_ENC = [0.9] * 12  # TO FREEZE, PUT 0
MOM_DEC = [0.9] * 12
WD_ENC = [1e-5] * 12  # TO FREEZE, PUT 0
WD_DEC = [1e-5] * 12
OPTIM_DEC = "sgd"
