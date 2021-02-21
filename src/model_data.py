import os
from model_config import *
IMAGE_IDS = os.listdir(os.path.join(PATH, 'Cover'))
N_IMAGES = len(IMAGE_IDS)