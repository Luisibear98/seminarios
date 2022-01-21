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

(x_train, _), (x_test, _) = fashion_mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

print (x_train.shape)
print (x_test.shape)


def create_encoder():
    input_imgs= Input(shape=(28,28,1), name="input_encoder")

    x = Conv2D(32, (3, 3), activation='relu', padding='same', name="conv1_encoder")(input_imgs) #Convolutional layer to get Features
    x = MaxPooling2D((2, 2), padding='same', name="MaxPooling_encoder")(x)                           #Reducing size of input
    x = Conv2D(32, (3, 3), activation='relu', padding='same', name="conv2_encoder")(x)          #Extracting more features
    encoded = MaxPooling2D((2, 2), padding='same', name="MaxPooling2_encoder")(x)                     #Getting the maximums

    encoder = Model(input_imgs, encoded)
    
    return encoder

def create_decoder():
    decoder_input= Input(shape=(7,7,32),name="input_decoder")

    decoder = Conv2D(32, (3, 3), activation='relu', padding='same',name="conv1_decoder")(decoder_input)
    x = UpSampling2D((2, 2),name="upsampling_decoder")(decoder)
    x = Conv2D(32, (3, 3), activation='relu', padding='same',name="Conv2_decoder")(x)
    x = UpSampling2D((2, 2),name="upsampling2_decoder")(x)
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same',name="conv2_decoder")(x)

    decoder = Model(decoder_input, decoded)

    return decoder


encoder = create_encoder()
plot_model(encoder, to_file='encoder.png')
decoder = create_decoder()
plot_model(decoder, to_file='decoder.png')
auto_input = Input(shape=(28,28,1))

encoded = encoder(auto_input)
decoded = decoder(encoded)

auto_encoder = Model(auto_input, decoded)

plot_model(auto_encoder, to_file='encoder-decoder.png')



callbacks = [TensorBoard(log_dir="./logs",
                         histogram_freq=1,
                         write_graph=True,
                         write_images=True,
                         update_freq='epoch',
                         profile_batch=2,
                         embeddings_freq=1)]



auto_encoder.compile(optimizer='adadelta', loss='binary_crossentropy')

auto_encoder.fit(x_train, x_train,
                epochs=1000,
                batch_size=2048,
                shuffle=True,
                validation_data=(x_test, x_test),
                callbacks=callbacks
               )

encoder.save_weights('./checkpoints_encoder/my_checkpoint_encoder')
decoder.save_weights('./checkpoints_decoder/my_checkpoint_decoder')

print(x_test[0])
encoded_imgs = encoder.predict(x_test)

decoded_imgs = decoder.predict(encoded_imgs)
print(decoded_imgs[0])