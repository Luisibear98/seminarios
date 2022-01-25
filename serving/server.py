from flask import Flask,request
from utils.auxiliar import initialize
import requests
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

app = Flask(__name__)


@app.route("/")
def init_server():

    return "welcome to this module"


def plotter(data):
    n = 10
    plt.figure(figsize=(20, 4))
    for i in range(1, n + 1):
    # Display original
       

    # Display reconstruction
        ax = plt.subplot(2, n, i + n)
        plt.imshow(data[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.savefig("output-rserver.png")


@app.route("/inference/",methods=['POST'])
def decrypt():

    data = request.json
    params = data['params']
    
    arr = np.array(data['arr'])
    decoded_imgs = decoder.predict(arr)
    plotter(decoded_imgs)
    

    return "done"

 


   
  


if __name__ == '__main__':

    decoder = initialize()
    app.run(host='0.0.0.0', port='1234')