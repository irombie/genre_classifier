#!/usr/bin/env python
# -*- coding: utf-8 -*-
import cv2
import numpy as np
import os
from PIL import Image
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense, Dropout
from keras import backend as K

class LeNet:
    @staticmethod
    def get_size_of_act_map(img_input, filt, stride, padding): #Using the formula (W−F+2P)/S+1, compute size of activation map
        inp_w = img_input.shape[0]
        inp_h = img_input.shape[1]
        depth = img_input.shape[2]
        filter_w = filt.shape[0]
        filter_h = filt.shape[1]
        out_w = (inp_w - filter_w + 2*padding)/stride + 1
        out_h = (inp_h - filter_h + 2*padding)/stride + 1
        return (out_w, out_h, depth)

    @staticmethod
    def build_model_lenet(img_h, img_w, depth, numClasses):
        model = Sequential()
        inp = (img_h, img_w, depth)
        #First convolutional layer: hyper parameters -> num_filters: 20, filter_size: 4x4, stride: 2x2, padding:1, bias: all zeroes, filter_init: Xavier uniform init
        model.add(Conv2D(6, (4,4), strides=(2, 2), input_shape=inp)) #Make the size of activation map integer using the formula (W−F+2P)/S+1
        model.add(Activation("relu")) #RELU layer
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2))) #Most common pooling layer type
        #second conv layer, all params are the except num_filters, the deeper the layer the more filters it uses
        model.add(Conv2D(10, (4, 4), strides=(2, 2),input_shape=inp))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        #flatten the output into a single vector before the first fully connected layer
        model.add(Flatten())
        model.add(Dropout(rate=0.1))
        model.add(Dense(500)) #found via trial and error (first fc layer)
        model.add(Activation("relu"))
        model.add(Dropout(rate=0.1))
        model.add(Dense(numClasses)) #second fc layer to output the probability of each data point belonging to each class
        model.add(Activation("softmax")) #using the prob. output of the last fc layer, classify to a class 
        return model




