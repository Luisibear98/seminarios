from unicodedata import name
from venv import create
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D


def create_decoder():
    decoder_input= Input(shape=(4,4,8),name="input_decoder")

   
    x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(decoder_input)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(16, (3, 3), activation='relu')(x)
    x = layers.UpSampling2D((2, 2))(x)
    decoded = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
    decoder = Model(decoder_input, decoded)

    return decoder


def initialize():
    
    decoder = create_decoder()
    dir = "/Users/luisibanezlissen/github/seminarios-main/checkpoints_decoder"
    latest = tf.train.latest_checkpoint(dir)

    decoder.load_weights(latest)

    return decoder 