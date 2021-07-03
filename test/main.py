
from keras.utils.io_utils import save_to_binary_h5py
import tensorflow as tf
import os
PATH = "D:/mary_study/"
EMBEDDING_ALGORYTHM='S-UNIWARD';
DATA_PATH = os.path.join(PATH, 'stego', EMBEDDING_ALGORYTHM)
#Embedding level - change it
LEVEL = 50

ALGORITHMS = [str(LEVEL)]
IMG_SIZE = 512
RGB = False
DROPOUT_RATE=0.1
EPOCHS = 50
BATCH_SIZE = 4
IMAGES_TO_PICK = 2000

TRAINING_TYPE = 'GPU' # TPU, GPU, CPU

prefix = f"drop_{DROPOUT_RATE}";
if DROPOUT_RATE == 0:
  prefix = "no_drop"



CHECKPOINT_PATH = PATH + f"checkpoints/training_{prefix}_{EMBEDDING_ALGORYTHM}_{LEVEL}_{IMAGES_TO_PICK}/cp-epoch-"+ "{epoch:04d}.ckpt"
LOAD_MODEL_PATH = PATH + f"models/training_{prefix}_{EMBEDDING_ALGORYTHM}_{LEVEL}_{IMAGES_TO_PICK}.h5"
RECOVERY_MODEL_PATH = PATH + f"models_recovery/training_{prefix}_{EMBEDDING_ALGORYTHM}_{LEVEL}_{IMAGES_TO_PICK}.h5"
MODEL_PATH = PATH + f"models/training_no_{prefix}_{EMBEDDING_ALGORYTHM}_{LEVEL}_{IMAGES_TO_PICK}.h5"
# os.listdir('/content/gdrive/MyDrive/Study_')
# IMAGE_IDS = os.listdir(os.path.join(DATA_PATH, 'Cover'))
# N_IMAGES = len(IMAGE_IDS)
import tensorflow.keras.layers as L
from tensorflow.keras import Model, metrics

def layer_type1(x_inp, filters, kernel_size=(3, 3), dropout_rate=0):
    x = L.Conv2D(filters, kernel_size, padding="same")(x_inp)
    x = L.BatchNormalization()(x)
    x = L.ReLU()(x)
    if dropout_rate > 0:
        x = L.Dropout(dropout_rate)(x)

    return x

def layer_type2(x_inp, filters, kernel_size=(3, 3), dropout_rate=0):
    x = layer_type1(x_inp, filters)
    x = L.Conv2D(filters, kernel_size, padding="same")(x)
    x = L.BatchNormalization()(x)
    x = L.ReLU()(x)
    if dropout_rate > 0:
        x = L.Dropout(dropout_rate)(x)

    x = L.Add()([x, x_inp])
    
    return x

def layer_type3(x_inp, filters, kernel_size=(3, 3), dropout_rate=0):
    x = layer_type1(x_inp, filters)
    x = L.Conv2D(filters, kernel_size, padding="same")(x)
    x = L.BatchNormalization()(x)
    x = L.ReLU()(x)
    x = L.AveragePooling2D(pool_size=(3, 3), strides=(2, 2))(x)
    if dropout_rate > 0:
        x = L.Dropout(dropout_rate)(x)
        
    x_res = L.Conv2D(filters, kernel_size, strides=(2, 2))(x_inp)
    x_res = L.BatchNormalization()(x_res)
    if dropout_rate > 0:
        x_res = L.Dropout(dropout_rate)(x_res)

    x = L.Add()([x, x_res])
    
    return x

def layer_type4(x_inp, filters, kernel_size=(3, 3), dropout_rate=0):
    x = layer_type1(x_inp, filters)
    x = L.Conv2D(filters, kernel_size=kernel_size, padding="same")(x)
    x = L.BatchNormalization()(x)
    if dropout_rate > 0:
        x = L.Dropout(dropout_rate)(x)    
    x = L.GlobalAveragePooling2D()(x)
    
    return x


def make_model(input_shape=(IMG_SIZE, IMG_SIZE, 3), num_type2=5, dropout_rate=0):
    # I reduced the size (image size, filters and depth) of the original network because it was way to big
    inputs = L.Input(shape=input_shape)
    
    x = layer_type1(inputs, filters=64, dropout_rate=dropout_rate)
    x = layer_type1(x, filters=16, dropout_rate=dropout_rate)    
    
    for _ in range(num_type2):
        x = layer_type2(x, filters=16, dropout_rate=dropout_rate)         
    
    x = layer_type3(x, filters=16, dropout_rate=dropout_rate) 
    x = layer_type3(x, filters=32, dropout_rate=dropout_rate)            
    x = layer_type3(x, filters=64, dropout_rate=dropout_rate)            
    #x = layer_type3(x, filters=128, dropout_rate=dropout_rate) 
    
    x = layer_type4(x, filters=128, dropout_rate=dropout_rate)        
    
    x = L.Dense(64)(x)
    x = L.BatchNormalization()(x)
    x = L.ReLU()(x)
    if dropout_rate > 0:
        x = L.Dropout(dropout_rate)(x)

    predictions = L.Dense(1, activation="sigmoid")(x)
    
    model = Model(inputs=inputs, outputs=predictions)
    # keras_auc = AUC()
    
    model.compile(optimizer='adamax',
                  loss='binary_crossentropy', 
                  metrics=[
                        metrics.MeanSquaredError(),
                        metrics.FalseNegatives(),
                        metrics.FalsePositives(),
                        metrics.TrueNegatives(),
                        metrics.TruePositives(),
                        metrics.AUC(),
                    ])
    
    return model


import numpy as np
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

def load_training_data_multi(n_images=100, algorithms = ALGORITHMS):
    train_data = []
    data_paths = [os.listdir(os.path.join(DATA_PATH, alg)) for alg in ['Cover'] + algorithms]
    labels = [np.zeros(N_IMAGES)]
    for _ in range(len(algorithms)):
        labels.append(np.ones(N_IMAGES))
    print('Loading...')
    for i, image_path in enumerate(data_paths):
        print(f'\t {i+1}-th folder')
        
        train_data_alg = joblib.Parallel(n_jobs=4, backend='threading')(
            joblib.delayed(load_image)([i, j, os.path.join(DATA_PATH, [['Cover'] + algorithms][0][i], img_p), labels]) for j, img_p in enumerate(image_path[:n_images]))

        train_data.extend(train_data_alg)
        
    shuffle(train_data)
    return train_data


import warnings

import numpy as np
import tensorflow as tf
from time import time
from sklearn.model_selection import train_test_split
import gc
warnings.filterwarnings("ignore")

heavy_memory_storage = []

def main():
    model = None
    try:
      # Seed for make things reproducable
      seed_everything()

      # Create model
    #   if (os.path.exists(LOAD_MODEL_PATH)):
    #     from keras.models import load_model
    #     print(f"Loading model from ${LOAD_MODEL_PATH}")
    #     model = load_model(LOAD_MODEL_PATH)
    #   else:
    #     model = make_model(input_shape=(IMG_SIZE, IMG_SIZE, 1), num_type2=4, dropout_rate=DROPOUT_RATE)
      model = make_model(input_shape=(IMG_SIZE, IMG_SIZE, 1), num_type2=4, dropout_rate=DROPOUT_RATE)
      # Save model graph to file
      tf.compat.v1.keras.utils.plot_model(model, show_shapes=True, to_file= PATH + "model.png")

      # start = time()
      # Get training data
      # train_data = get_train_data()
      # Clear RAM unused data
      # clear_heavy_memory_storage()
      # min_elapsed = (time() - start) / 60
      # print("{0:.2f} min elapsed".format(min_elapsed))

      train_model(model)
      print(f"Saving model to {MODEL_PATH}")
      model.save(MODEL_PATH)
    except Exception as err:
      print(err)
      if (model is not None):
        print(f"Saving model to {MODEL_PATH}")
        model.save(MODEL_PATH)  

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
    training_data = load_training_data_multi(n_images=IMAGES_TO_PICK)
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

def load_image_1(data):
    i, j, img_path, labels = data
    
    img = Image.open(img_path)
    if RGB:
        img = img.convert('RGB')
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.ANTIALIAS)
    label = labels[i][j]
    return [np.array(img), label]

def load_image_for_gen(filename):
    return np.array(Image.open(filename).resize((IMG_SIZE, IMG_SIZE), Image.ANTIALIAS))

max_length = 80002*2

def get_data_for_pgm_generator(images_count, images_start = 0):
    i = 0
    base_folder = 'C:/mary_study/training'
    stego_folder = os.path.join(base_folder, '50')
    cover_folder = os.path.join(base_folder, 'Cover')
    stego_image_filenames = [os.path.join(stego_folder, x) for x in os.listdir(stego_folder)[images_start:images_start+images_count]]
    cover_image_filenames = [os.path.join(cover_folder, x) for x in os.listdir(cover_folder)[images_start:images_start+images_count]]
    filenames = []
    labels = {}
    for filename in stego_image_filenames:
        filenames.append(filename)
        labels[filename] = 1
    for filename in cover_image_filenames:
        filenames.append(filename)
        labels[filename] = 0
    return filenames, labels

def image_generator():
    i = 0
    base_folder = 'C:/mary_study/training'
    stego_folder = os.path.join(base_folder, '50')
    cover_folder = os.path.join(base_folder, 'Cover')
    stego_image_filenames = [os.path.join(stego_folder, x) for x in os.listdir(stego_folder)]
    cover_image_filenames = [os.path.join(cover_folder, x) for x in os.listdir(cover_folder)]
    filenames_and_labels = []
    for filename in stego_image_filenames:
        filenames_and_labels.append([filename, 1])
    for filename in cover_image_filenames:
        filenames_and_labels.append([filename, 0])
    shuffle(filenames_and_labels)
    while i < max_length:
        next_result = filenames_and_labels[i]
        yield  next_result[1], load_image_for_gen(next_result[0])
        i+=1

def train_model(model):
    from generators import PgmDataGenerator
    params = {
        'dimension': (IMG_SIZE, IMG_SIZE),
        'batch_size': BATCH_SIZE,
        'n_classes': 2,
        'n_channels': 1,
        'shuffle': True
    }
    start = time()
    train_filenames, train_labels = get_data_for_pgm_generator(2000)
    val_filenames, val_labels = get_data_for_pgm_generator(2000, 2000)
    training_generator = PgmDataGenerator(train_filenames, train_labels, **params)
    val_generator = PgmDataGenerator(val_filenames, val_labels, **params)

    save_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=CHECKPOINT_PATH,
            verbose=1,
            save_weights_only=True
        )
    model.fit(training_generator, validation_data = val_generator, epochs = EPOCHS, callbacks = [save_callback])
    # model.fit_generator(generator = training_generator,
    #                     validation_data = val_generator,
    #                     use_multiprocessing = True,
    #                     workers = 6)

    
    model.save_weights(CHECKPOINT_PATH.format(epoch=0))
   
    min_elapsed = (time() - start) / 60
    print("{0:.2f} min elapsed for training".format(min_elapsed))
def load_weights_if_exists(model):
    checkpoint_dir = os.path.dirname(CHECKPOINT_PATH)
    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
    if latest_checkpoint is not None:
        print(f"Found a checkpoint {latest_checkpoint}. Loading...")
        model.load_weights(latest_checkpoint)
    else:
        print("No checkpoints found, starting from untrained model...")
with tf.device('/device:GPU:0'):
    main()
# https://keras.io/api/preprocessing/image/

# from tensorflow.keras.preprocessing.image import ImageDataGenerator

# from model_config import IMG_SIZE


# DATA_PATH = 'D:/mary_study/test_2k'



# datagen = ImageDataGenerator(
#     dtype='int',
#     validation_split = 0.25
# )

# generator = datagen.flow_from_directory(
#     DATA_PATH,
#     target_size=(512,512),
#     color_mode='grayscale',
#     class_mode='binary' ,
#     batch_size=64,
#     follow_links=True
# )
# model = make_model(input_shape=(IMG_SIZE, IMG_SIZE, 1))

# with tf.device('/device:GPU:0'):
#     model.fit(
#         generator,
#         steps_per_epoch=500,
#         epochs=50
#         )


