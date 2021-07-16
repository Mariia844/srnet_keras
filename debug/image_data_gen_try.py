from model import make_model
import os

from tensorflow.python.ops.gen_array_ops import batch_to_space
import generators
IMG_SIZE = 28
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 1)
BATCH_SIZE = 32
BATCHED_IMAGE_SIZE = (BATCH_SIZE, IMG_SIZE, IMG_SIZE, 1)
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.optimizers import SGD
from sklearn.model_selection import KFold
from tensorflow.keras.preprocessing.image import img_to_array

INPUT_SIZE = (256, 256)

def define_model():
    return make_model(input_shape=(*INPUT_SIZE, 1), dropout_rate=0.1)

# def evaluate_model(, n_splits = 5):
#     scores, histories = list(), list()
#     kfold = KFold(n_splits=n_splits, shuffle=True, random_state=1)
#     for train_ix, test_ix in kfold.

def main():
    image_directory = 'E:/Mary/training_bmp'
    params = {
        'labels': 'inferred',
        'label_mode': 'int',
        'color_mode': 'grayscale',
        'batch_size': 16,
        'image_size': INPUT_SIZE,
        'shuffle': True,
        'seed': 137,
        'validation_split': 0.5,
        'subset': 'training',
        'interpolation': "bilinear",
        'follow_links': False,
    }
    train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        image_directory,
        **params
    )
    validation_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        image_directory,
        **params
    )
    model = define_model()

    # tf.compat.v1.keras.utils.plot_model(model, show_shapes=True, to_file= "model.png")
    model.summary()
    
    model.fit(
        x=train_dataset, validation_data=validation_dataset, epochs=10)

    # model.save('digits_cnn.h5')

main()
# benchmark()