from flask import Flask
from utils.auxiliar import initialize


app = Flask(__name__)


@app.route("/")
def init_server():

    return "welcome to this module"

@app.route("/inference/<data>")
def decrypt(vendor):


   return 




if __name__ == '__main__':

    decoder = initialize()
    app.run()