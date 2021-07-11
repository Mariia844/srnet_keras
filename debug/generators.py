from PIL import Image
import glob, os, numpy as np
import tensorflow as tf
keras = tf.compat.v1.keras
# Sequence = keras.utils.Sequence

class PgmDataGenerator(keras.utils.Sequence):
    def __init__(self, paths, labels, batch_size=32, dimension=(512,512), n_channels=1,
                n_classes=2, shuffle=True):
        'Initialization'
        self.dimension = dimension
        self.batch_size = batch_size
        self.labels = labels
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.paths = paths
        self.on_epoch_end()
        # super().__init__()

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.paths))
        if (self.shuffle == True):
            np.random.shuffle(self.indexes)
    def __data_generation(self, images_names_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dimension, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        for i, name in enumerate(images_names_temp):
            try:
                arr = np.array(Image.open(name))
                resized_arr = np.expand_dims(arr, axis=0)
                # Fix only for our dataset
                if (resized_arr.shape[0] == 1):
                    resized_arr = np.expand_dims(arr, axis=2)
                X[i] = resized_arr
                y[i] = self.labels[name]
            except Exception as e:
                print(e)
        X = np.concatenate(X, axis=0)
        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.paths) / self.batch_size))
    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        paths_temp = [self.paths[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(paths_temp)

        return X, y
    