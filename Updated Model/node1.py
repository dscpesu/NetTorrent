import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import sklearn

from sklearn.metrics import accuracy_score as acc_score

import tensorflow as tf

import keras

from keras import backend as K

from keras.models import model_from_json

from keras.models import load_model

import concurrent.futures

from tensorflow import Graph , Session

from keras.datasets import cifar10

from keras.preprocessing.image import ImageDataGenerator

from keras.layers import Dense, Dropout, Activation, Flatten

from keras.layers import Conv2D, MaxPooling2D

import os

import requests

import flask

import threading

from keras import Sequential


import time

import math

app = flask.Flask(__name__)

thread_local = threading.local()
batch_size = 32
num_classes = 10
epochs = 100
data_augmentation = True
num_predictions = 20
# The data, split between train and test sets:
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)[:25000]
# y_test = keras.utils.to_categorical(y_test, num_classes)

global model , accuracy  , rec_acc , graph , n_iters , count_iter

rec_acc = None

graph = Graph()

with graph.as_default():

    sess = Session()

K.set_session(sess)

with graph.as_default():

    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same',input_shape=x_train.shape[1:] , activation = 'relu'))
    model.add(Conv2D(32, (3, 3) , activation = 'relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same', activation = 'relu'))
    model.add(Conv2D(64, (3, 3), activation = 'relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512 ,  activation = 'relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes , activation = 'softmax'))

    # initiate RMSprop optimizer

    # Let's train the model using RMSprop
    model.compile(loss='categorical_crossentropy',
                optimizer='rmsprop',
                metrics=['accuracy'])

x_train = x_train.astype('float32')[:25000]
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

@app.route('/api/node1/start', methods = ['POST'])
def start():

    for i in range(100):

        for j in range(1):
        
            fit(i,j)

    return flask.Response(status = 200)

def fit(i,j):

    global count_iter , accuracy , rec_acc , graph , sess , model

    K.set_session(sess)

    with graph.as_default():

        model.fit(x_train,y_train,epochs = 1 , batch_size = 32)

        accuracy = get_accuracy()

        # model_json = model.to_json()

        # with open("model1.json", "w") as json_file:

        #     json_file.write(model_json)

        model.save_weights("model1.h5")

    requests.post("http://127.0.0.1:4000/api/node2/assign_acc",json={'acc':float(accuracy)})

    while rec_acc is None :

        pass  

    rec_acc = None

@app.route('/api/node1/assign_acc', methods = ['POST'])
def assign_accuracy():

    global model , rec_acc , sess , graph , opt

    rec_acc = flask.request.json['acc']

    if  (rec_acc > accuracy):

        print('REC')

        K.set_session(sess)

        with graph.as_default():
    
            # json_file = open('model2.json', 'r')

            # loaded_model_json = json_file.read()

            # json_file.close()

            # model = model_from_json(loaded_model_json)

            model.load_weights("model2.h5")

            model.compile(loss='categorical_crossentropy',
                        optimizer='rmsprop',
                        metrics=['accuracy'])
    return flask.Response(status = 200)



def get_accuracy():

    K.set_session(sess)

    with graph.as_default():

        y_pred = model.predict(x_test)
        
    y_pred = [np.argmax(i) for i in y_pred]

    accuracy = acc_score(y_test, y_pred)

    print(accuracy)

    return accuracy

accuracy = get_accuracy()

if __name__ == '__main__':

    app.run(host='127.0.0.1', port=2000)
