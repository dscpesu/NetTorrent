import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import sklearn

from sklearn.metrics import accuracy_score as acc_score

import tensorflow as tf

import keras

from keras.layers import Dense

from keras.datasets import mnist

from keras import backend as K

from keras.models import model_from_json

from keras.models import load_model

import concurrent.futures

from tensorflow import Graph , Session

import requests

import flask

import threading

from keras import Sequential


import time

import math

app = flask.Flask(__name__)

thread_local = threading.local()

(X_train,y_train),(X_test,y_test) = mnist.load_data()

X_train_flat = X_train.reshape((X_train.shape[0],-1))[30000:]

X_test_flat  = X_test.reshape((X_test.shape[0],-1))

y_train_oh = keras.utils.to_categorical(y_train,10)[30000:]

global model , accuracy  , rec_acc , graph , n_iters , count_iter

rec_acc = None

graph = Graph()

with graph.as_default():

    sess = Session()

K.set_session(sess)

with graph.as_default():

    model = Sequential()

    model.add(Dense(input_dim = 784 , units = 256 , activation = 'sigmoid'))

    model.add(Dense(units = 256 , activation = 'sigmoid'))

    model.add(Dense(units = 256 , activation = 'sigmoid'))

    model.add(Dense(units = 256 , activation = 'sigmoid'))

    model.add(Dense(units = 256 , activation = 'sigmoid'))

    model.add(Dense(units = 256 , activation = 'sigmoid'))

    model.add(Dense(units = 10 , activation = 'sigmoid'))

    model.compile(optimizer = 'rmsprop' , loss = 'categorical_crossentropy' , metrics = ['accuracy'])

@app.route('/api/node2/start', methods = ['POST'])
def start():

    for i in range(10):

        for j in range(1):
        
            fit(i,j)

    return flask.Response(status = 200)

def fit(i,j):

    global count_iter , accuracy , rec_acc , graph , sess , model

    K.set_session(sess)

    with graph.as_default():

        model.fit(X_train_flat[30000*j:30000*j + 30000],y_train_oh[30000*j:30000*j + 30000],epochs = 1 , batch_size = 1)

        accuracy = get_accuracy()

        model_json = model.to_json()

        with open("model2.json", "w") as json_file:

            json_file.write(model_json)

        model.save_weights("model2.h5")

    requests.post("http://127.0.0.1:2000/api/node1/assign_acc",json={'acc':float(accuracy)})

    while rec_acc is None :

        pass  

    rec_acc = None

@app.route('/api/node2/assign_acc', methods = ['POST'])
def assign_accuracy():

    global model , rec_acc , sess , graph

    rec_acc = flask.request.json['acc']

    if  (rec_acc > accuracy):

        print('REC')

        K.set_session(sess)

        with graph.as_default():
    
            json_file = open('model1.json', 'r')

            loaded_model_json = json_file.read()

            json_file.close()

            model = model_from_json(loaded_model_json)

            model.load_weights("model1.h5")

            model.compile(optimizer = 'rmsprop' , loss = 'categorical_crossentropy' , metrics = ['accuracy'])

    return flask.Response(status = 200)



def get_accuracy():

    K.set_session(sess)

    with graph.as_default():

        y_pred = model.predict(X_test_flat)
        
    y_pred = [np.argmax(i) for i in y_pred]

    accuracy = acc_score(y_test, y_pred)

    return accuracy

accuracy = get_accuracy()

if __name__ == '__main__':

    app.run(host='127.0.0.1', port=4000)