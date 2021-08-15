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

import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras.callbacks import ModelCheckpoint


import telebot
from timeit import default_timer as timer

INPUT_SIZE = (256, 256)
EPOCHS = 20
BASE_PATH = 'E:/Mary/history'
LAST_MODEL = 'E:/Mary/history/Training_mipod_3_png_05_08_2021_21_50_26/saved-model-ep_20-loss_0.33-val_loss_0.65.hdf5'
LOAD_MODEL = False
HISTORY_NAME = 'Training_suni_10'
bot = telebot.TeleBot('1292996210:AAFVuU6mo6-rS2Tv6Xuy3ZucfdJzTaBN9PY')
CHAT_ID = 337882617
def define_model():
    return make_model(input_shape=(*INPUT_SIZE, 1), dropout_rate=0.1)

def evaluate_model(dataset, n_splits = 5):
    scores, histories = list(), list()
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=1)
    for train_ix, test_ix in kfold.split(dataset):
        pass

def main():
    image_directory = 'E:/Mary/1.jpg_data'
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
    start = timer()
    train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        image_directory,
        **params
    )
    validation_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        image_directory,
        **params
    )
    model = define_model()
    if LOAD_MODEL:
        from keras.models import load_model
        print(f'Loading model from {LAST_MODEL}')
        model = load_model(LAST_MODEL)
    from datetime import datetime

    today = datetime.now()

    # dd/mm/YY
    d1 = today.strftime("%d_%m_%Y_%H_%M_%S")
    dir_to_create = os.path.join(BASE_PATH, f"{HISTORY_NAME}_{d1}")
    os.makedirs(dir_to_create)
    filepath = dir_to_create+ "/saved-model-ep_{epoch:02d}-loss_{loss:.2f}-val_loss_{val_loss:.2f}.hdf5"
    hist_csv_file = f"{dir_to_create}/history.csv" 
    checkpoint = ModelCheckpoint(filepath, monitor='accuracy', verbose=1,
        save_best_only=False, mode='auto', period=1)

    # tf.compat.v1.keras.utils.plot_model(model, show_shapes=True, to_file= "model.png")
    # model.summary()
    history = model.fit(
        x=train_dataset, 
        validation_data=validation_dataset, 
        epochs=EPOCHS,
        callbacks=[checkpoint])
    hist_df = pd.DataFrame(history.history) 
    with open(hist_csv_file, mode='w') as f:
        hist_df.to_csv(f)
    plt.plot(history.history['auc'])
    plt.plot(history.history['val_auc'])
    plt.title('model AUC')
    plt.ylabel('auc')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    auc_path = os.path.join(dir_to_create, 'auc.png')
    plt.savefig(auc_path)
    plt.clf()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    loss_path = os.path.join(dir_to_create, 'loss.png')
    plt.savefig(loss_path)
    end = timer()
    bot.send_message(chat_id=CHAT_ID, text=f'training \"{HISTORY_NAME}\" completed. {start - end}s elapsed')
    #bot.send_document(chat_id=CHAT_ID, text='History', )
    with open(hist_csv_file) as hist:
        bot.send_document(chat_id=CHAT_ID, data=hist)
    with open(auc_path, 'rb') as auc:
        bot.send_photo(chat_id=CHAT_ID, photo=auc)
    with open(loss_path, 'rb') as loss:
        bot.send_photo(chat_id=CHAT_ID, photo=loss)

main()
# benchmark()