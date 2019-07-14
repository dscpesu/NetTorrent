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

X_train_flat = X_train.reshape((X_train.shape[0],-1))

X_test_flat  = X_test.reshape((X_test.shape[0],-1))

y_train_oh = keras.utils.to_categorical(y_train,10)

global model

global accuracy , rec_acc

model = Sequential()

model.add(Dense(input_dim = 784 , units = 256 , activation = 'sigmoid'))

model.add(Dense(units = 128 , activation = 'sigmoid'))

model.add(Dense(units = 10 , activation = 'sigmoid'))

model.compile(optimizer = 'adam' , loss = 'binary_crossentropy' , metrics = ['accuracy'])

fit(X_train_flat , y_train_oh , X_test_flat , y_test , batch_size = 100 , epochs = 10 , step = 10)

model.fit(X_train , y_train , epochs = 1 , batch_size = int(batch_size))


def fit(X_train , y_train , X_test , y_test , step , epochs = 1 , batch_size = Non):
        
    global model , accuracy , rec_acc

    X_train_split = np.array([np.split(_,self.steps) for _ in X_train])
        
    y_train_split = np.array([np.split(_,self.steps) for _ in y_train])

    for i in range(epochs):

        for j in range(step):

            model.fit(X_train_split[j] , y_train_split[j] , epochs = 1 , batch_size = int(batch_size))

            accuracy = test_acc(X_test , y_test)

            assign_acc()

            while not check:

                rec_acc = get_acc()

            if accuracy <= rec_acc :

                while not check :

                    pass  
                
                get_model()
            
            else:

                assign_model()

            check = False
    
@app.route('/api/node2/switch', methods = ['POST'])
def switchCheck():

    global check

    check = not check

@app.route('/api/node2/get', methods = ['POST'])
def get_acc():

    global rec_acc

    rec_acc = flask.request.json['acc']

    switchCheck()

def assign_acc():

    global accuracy 

    # send accuracy to node 1

    # call switchCheck in node 1


def get_model():

    global model

    json_file = open('model.json', 'r')

    loaded_model_json = json_file.read()

    json_file.close()

    loaded_model = model_from_json(loaded_model_json)

    loaded_model.load_weights("model.h5")

    model = loaded_model

    model.compile(optimizer = 'adam' , loss = 'binary_crossentropy' , metrics = ['accuracy'])


def assign_model():

    model_json = model.to_json()

    with open("model.json", "w") as json_file:

        json_file.write(model_json)

    model.save_weights("model.h5")

    # call switchCheck in node 1


def test_acc(X_test , y_test):

    global model

    y_pred = model.predict(X_test)
        
    y_pred = [np.argmax(i) for i in y_pred]

    accuracy = acc_score(y_test, y_pred)

    return accuracy

if __name__ == '__main__':

    app.run(host='127.0.0.1', port=4000)