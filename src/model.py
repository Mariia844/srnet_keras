import keras.layers as L
from keras import Model, metrics
import os
from model_config import *
from model_data import *
# from tensorflow.keras.metrics import AUC



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
    x = L.Conv2D(filters, kernel_size=(3, 3), padding="same")(x)
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
    
    model.compile(optimizer='adam',
                  loss='binary_crossentropy', 
                  metrics=[
                        metrics.MeanSquaredError(),
                        metrics.AUC()
                    ])
    
    return model