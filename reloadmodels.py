from unicodedata import name
from venv import create
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.utils import plot_model

from tensorflow.keras.callbacks import TensorBoard

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Model
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D



from keras.datasets import mnist

(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))



def create_encoder():
    input_imgs= Input(shape=(28,28,1), name="input_encoder")
    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(input_imgs)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    encoded = layers.MaxPooling2D((2, 2), padding='same')(x)

    encoder = Model(input_imgs, encoded)
    
    return encoder

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


encoder = create_encoder()
decoder = create_decoder()
latest = tf.train.latest_checkpoint("/home/oso/git/Seminarios/checkpoints_encoder/")
encoder.load_weights(latest)
latest = tf.train.latest_checkpoint("/home/oso/git/Seminarios/checkpoints_decoder/")

decoder.load_weights(latest)


print(x_test[0])
encoded_imgs = encoder.predict(x_test)

decoded_imgs = decoder.predict(encoded_imgs)


n = 10
plt.figure(figsize=(20, 4))
for i in range(1, n + 1):
    # Display original
    ax = plt.subplot(2, n, i)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Display reconstruction
    ax = plt.subplot(2, n, i + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.savefig("output-reloaded.png")
