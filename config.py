import logging
from easydict import EasyDict as edict
from math import log2


__C = edict()
# for consumers
cfg = __C
# NAMES
__C.PROJECT_NAME = "ProGAN"
__C.PROJECT_VERSION_NAME = "ProGAN default"
__C.DATASET_NAME = "AnimeFaces_256px"
# Global
__C.LEARNING_RATE = 1e-3
__C.BATCH_SIZE = [128, 128, 128, 64, 32, 16, 8, 4, 4]
# Gradient penalty
__C.CRITIC_ITERATIONS = 1
__C.LAMBDA_GP = 10
# SHAPES
__C.START_TRAIN_IMG_SIZE = 128    # we start from 4x4 images
__C.IMG_SIZE = 256
__C.CHANNELS_IMG = 3
__C.Z_DIMENSION = 256
# Models features
__C.IN_CHANNELS = 256
# Grow
__C.NUM_STEPS = int(log2(__C.IMG_SIZE / 4)) + 1
__C.PROGRESSIVE_EPOCHS = [30] * len(__C.BATCH_SIZE)
# Paths and saves
__C.SAVE_MODEL = True
__C.OUT_DIR = ''
__C.PATH_TO_LOG_FILE = 'ProGAN/data/logs/train.log'
# Display results
__C.NUM_SAMPLES = 4            # image grid shape <- sqrt(NUM_SAMPLES)
__C.FREQ = 100
__C.RESUME_ID = None
# init logs
logger = logging.getLogger()
c_handler = logging.StreamHandler()
f_handler = logging.FileHandler(__C.PATH_TO_LOG_FILE, mode='a', encoding='utf-8')

c_handler.setLevel(logging.INFO)
f_handler.setLevel(logging.ERROR)
c_format = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
f_format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
c_handler.setFormatter(c_format)
f_handler.setFormatter(f_format)
logger.addHandler(c_handler)
logger.addHandler(f_handler)
logger.setLevel(logging.INFO)
