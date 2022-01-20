from venv import create
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Model
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D

(x_train, _), (x_test, _) = fashion_mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

print (x_train.shape)
print (x_test.shape)


def create_encoder():
    input_imgs= Input(shape=(28,28,1))

    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_imgs) #Convolutional layer to get Features
    x = MaxPooling2D((2, 2), padding='same')(x)                           #Reducing size of input
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)          #Extracting more features
    encoded = MaxPooling2D((2, 2), padding='same')(x)                     #Getting the maximums

    encoder = Model(input_imgs, encoded)
    
    return encoder

def create_decoder():
    decoder_input= Input(shape=(7,7,32))

    decoder = Conv2D(32, (3, 3), activation='relu', padding='same')(decoder_input)
    x = UpSampling2D((2, 2))(decoder)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

    decoder = Model(decoder_input, decoded)

    return decoder


encoder = create_encoder()
decoder = create_decoder()

auto_input = Input(shape=(28,28,1))

encoded = encoder(auto_input)
decoded = decoder(encoded)

auto_encoder = Model(auto_input, decoded)


auto_encoder.compile(optimizer='adadelta', loss='binary_crossentropy')

auto_encoder.fit(x_train, x_train,
                epochs=1000,
                batch_size=2048,
                shuffle=True,
                validation_data=(x_test, x_test),
                #callbacks=[TensorBoard(log_dir='/tmp/tb', histogram_freq=0, write_graph=False)]
               )