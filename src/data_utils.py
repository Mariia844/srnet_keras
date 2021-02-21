from model_config import *
from model_data import *
import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import roc_curve, auc
# import tensorflow as tf
from PIL import Image
import joblib
from random import shuffle


def load_image(data):
    i, j, img_path, labels = data
    
    img = Image.open(img_path)
    if RGB:
        img = img.convert('RGB')
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.ANTIALIAS)
    label = labels[i][j]
    return [np.array(img), label]

def load_training_data_multi(n_images=100):
    train_data = []
    data_paths = [os.listdir(os.path.join(PATH, alg)) for alg in ['Cover'] + ALGORITHMS]
    labels = [np.zeros(N_IMAGES)]
    for _ in range(len(ALGORITHMS)):
        labels.append(np.ones(N_IMAGES))
    print('Loading...')
    for i, image_path in enumerate(data_paths):
        print(f'\t {i+1}-th folder')
        
        train_data_alg = joblib.Parallel(n_jobs=4, backend='threading')(
            joblib.delayed(load_image)([i, j, os.path.join(PATH, [['Cover'] + ALGORITHMS][0][i], img_p), labels]) for j, img_p in enumerate(image_path[:n_images]))

        train_data.extend(train_data_alg)
        
    shuffle(train_data)
    return train_data