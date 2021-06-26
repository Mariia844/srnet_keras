import os
from model_config import *
IMAGE_IDS = os.listdir(os.path.join(PATH, '0'))
N_IMAGES = len(IMAGE_IDS)