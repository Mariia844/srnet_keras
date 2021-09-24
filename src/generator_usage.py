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
    image = Image.open(filename)
    #image = image.reshape((28, 28, 1))
    image = tf.image.resize(image, IMG_SHAPE)
    image = image / 255.0
    return image
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
    label = int(file_name.split('_')[0])
    label_array = np.zeros(10)
    label_array[label] = 1
    return label_array
def get_filename(tensor):
    return tensor.numpy().decode('utf-8')
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
def set_image_shapes(image, label):
    image.set_shape(IMG_SHAPE)
    return image, label


def define_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28,28, 1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(10, activation='softmax'))
	# compile model
    opt = SGD(lr=0.01, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# def evaluate_model(, n_splits = 5):
#     scores, histories = list(), list()
#     kfold = KFold(n_splits=n_splits, shuffle=True, random_state=1)
#     for train_ix, test_ix in kfold.

def main():
    test_pattern = 'D:/dev/github/YunYang1994/yymnist/mnist/test/*.pgm'
    test_dataset = tf.data.Dataset.list_files(test_pattern)
    train_pattern = 'D:/dev/github/YunYang1994/yymnist/mnist/train/*.pgm'
    train_dataset = tf.data.Dataset.list_files(train_pattern)

    test_image_dataset = test_dataset.map(
        map_func=dataset_map,
        num_parallel_calls=tf.data.experimental.AUTOTUNE).map(set_image_shapes)
    train_image_dataset = train_dataset.map(
        map_func=dataset_map,
        num_parallel_calls=tf.data.experimental.AUTOTUNE).map(set_image_shapes)
    model = define_model()
    # tf.compat.v1.keras.utils.plot_model(model, show_shapes=True, to_file= "model.png")
    model.summary()
    # for image,label in test_image_dataset.take(10):
    #     print(label, ' ', image.shape)
    # model.fit(train_image_dataset, epochs=10, batch_size=32, validation_data=test_image_dataset)  
    a = np.arange(1, 10)
    labels = list()
    # for i in range(1000):
    #     choice = np.random.choice(a)
    #     labels_i = np.zeros(10)
    #     labels_i[choice] = 1
    #     labels.append(labels_i)
    
    # labels = np.array(labels)
#     model.fit(
#     np.random.randn(1000, 28, 28, 1),
#     labels,
#     epochs=10,
#     steps_per_epoch=2000//BATCH_SIZE,
#     validation_data=(np.random.randn(1000, 28, 28, 1),np.ones(1000)),
#     batch_size=BATCH_SIZE,
#     verbose=1,
#     # callbacks=CALLBACKS,
#     workers=1,
#     use_multiprocessing=True
# )
    # for image, label in train_image_dataset.batch(BATCH_SIZE).take(1):
    #     print(image.shape, label)
    # train_image_dataset = train_image_dataset.batch(BATCH_SIZE)
    # # train_image_dataset = train_image_dataset.repeat()
    # test_image_dataset = test_image_dataset.batch(BATCH_SIZE)
    # test_image_dataset = test_image_dataset.repeat()
    model.fit(x=train_image_dataset.make_one_shot_iterator(), validation_data=test_image_dataset.make_one_shot_iterator(), epochs=10)

    # model.save('digits_cnn.h5')

main()
# benchmark()