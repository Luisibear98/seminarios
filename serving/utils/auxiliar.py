
import tensorflow as tf
from tensorflow.keras.models import Model

def initilalize():

    decoder = Model()
    latest = tf.train.latest_checkpoint("/home/oso/git/Seminarios/checkpoints_decoder/")
    decoder.load_weights(latest)


    
    
    return decoder 