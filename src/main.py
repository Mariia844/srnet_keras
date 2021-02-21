# Basic imports
import os

import warnings

import numpy as np
import tensorflow as tf

from model import make_model
from time import time
from model_config import *

from data_utils import load_training_data_multi
from sklearn.model_selection import train_test_split
import gc
from utils import save_model

warnings.filterwarnings("ignore")

heavy_memory_storage = []



def main():
    # Seed for make things reproducable
    seed_everything()

    # Create model
    model = make_model(input_shape=(IMG_SIZE, IMG_SIZE, 3 if RGB else 1), num_type2=4)
    # Save model graph to file
    tf.compat.v1.keras.utils.plot_model(model, show_shapes=True, to_file="model.png")

    start = time()
    # Get training data
    train_data = get_train_data()
    # Clear RAM unused data
    clear_heavy_memory_storage()
    print(f"{(time() - start) / 60: .2f} min elapsed.")

    train_model(model, train_data)

    save_model(model)

# To make things reproductible

def seed_everything(seed=42):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_KERAS'] = '1'
def set_cpu_count():
    cpu_str = str(CPU_COUNT)
    os.environ['MKL_NUM_THREADS'] = cpu_str
    os.environ['GOTO_NUM_THREADS'] = cpu_str
    os.environ['OMP_NUM_THREADS'] = cpu_str
    os.eviron['openmp'] = 'True'
def get_train_data():
    global heavy_memory_storage
    training_data = load_training_data_multi(n_images=5000)
    channels_count_in_image = 3 if RGB else 1
    trainImages = np.array([i[0] for i in training_data]).reshape(-1, IMG_SIZE, IMG_SIZE, channels_count_in_image)
    trainLabels = np.array([i[1] for i in training_data], dtype=int)
    heavy_memory_storage.append(training_data)
    heavy_memory_storage.append(trainLabels)
    heavy_memory_storage.append(trainImages)
    return train_test_split(trainImages, trainLabels, random_state=42, stratify=trainLabels)


def clear_heavy_memory_storage():
    global heavy_memory_storage
    while (len(heavy_memory_storage) > 0):
        el = heavy_memory_storage.pop()
        del el
    gc.collect()

def train_model(model, train_data):
    # tf.compat.v1.enable_eager_execution()
    X_train, X_val, y_train, y_val = train_data
    model.fit(X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS, 
        verbose=1)
    # config = tf.compat.v1.ConfigProto(device_count={"CPU": CPU_COUNT})
    # with tf.compat.v1.Session(config=config, graph=tf.compat.v1.get_default_graph()) as session:
    #     tf.compat.v1.keras.backend.set_session(session=session)
       


if __name__ == "__main__":
    main()