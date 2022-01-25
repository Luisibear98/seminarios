from unicodedata import name
from venv import create
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Model
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.datasets import mnist
import requests
'''
Loading data
'''
(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))


'''
Creating encoder
'''
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



encoder = create_encoder()


'''
Loading trained models
'''
latest = tf.train.latest_checkpoint("/Users/luisibanezlissen/github/seminarios-main/checkpoints_encoder")
encoder.load_weights(latest)



print(x_test[0])
decoded_imgs = encoder.predict(x_test)

url =  "http://127.0.0.1:1234/inference/"



params = {'param0': 'param0'}

data = {'params': params, 'arr': decoded_imgs.tolist()}
print(data)
response = requests.post(url, json=data)
print(response)

'''
Example functional working


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
'''