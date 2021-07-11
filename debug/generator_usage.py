import os
import generators
IMG_SIZE = 28
BATCH_SIZE = 32

import tensorflow as tf
import numpy as np
from PIL import Image

def get_data_for_pgm_generator(images_count, images_start = 0):
    i = 0
    base_folder = 'D:/dev/github/YunYang1994/yymnist/mnist/test'
    image_filenames  = [os.path.join(base_folder, x) for x in os.listdir(base_folder)[images_start:images_start+images_count]]
    filenames = []
    labels = {}
    for filename in image_filenames:
        filenames.append(filename)
        labels[filename] = filename.split('_')[0]
    return filenames, labels

def load_image(filename):
    return np.array(Image.open(filename)).astype('int32')
def load_image_and_label(filename):
    return load_image(filename)
def load_image_tensor(tensor):
    filename = tensor.numpy().decode('utf-8')
    return load_image(filename)
def load_data_tensor(tensor):
    filename = tensor.numpy().decode('utf-8')
    return load_image_and_label(filename)
def get_label(tensor):
    filename = tensor.numpy().decode('utf-8')
    head, file_name = os.path.split(filename)
    return int(file_name.split('_')[0])

def dataset_map(tensor):
    label = tf.py_function(
        func=get_label,
        inp=[tensor],
        Tout=[tf.int32])
    image = tf.py_function(
        func=load_image_tensor,
        inp=[tensor],
        Tout=[tf.int32])
    return image, label[0]

def main():
    pattern = 'D:/dev/github/YunYang1994/yymnist/mnist/test/*.pgm'
    dataset = tf.data.Dataset.list_files(pattern)
    for filename in dataset.take(10):
        print(filename.numpy().decode('utf-8'))
        # tensor = tf.convert_to_tensor(load_image(filename.numpy().decode('utf-8')))
        # print(tensor.numpy().shape)
        #print(tensor.shape)
    # image_dataset = dataset.map(lambda x : tf.py_function(load_image, [x], [tf.string]))
    # for image in image_dataset.take(10):
    #     print(image.shape)
    image_dataset = dataset.map(
        map_func=dataset_map,
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
    for image, label in image_dataset.take(10):
        print(label, image.shape)
    #     pass
image = 'D:/dev/github/YunYang1994/yymnist/mnist/test/0_00003.pgm'
arr = load_image(image)
from tensorflow.python.framework import ops
main()
# benchmark()