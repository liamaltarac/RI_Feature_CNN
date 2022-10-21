import numpy as np
import tensorflow as tf
from tensorflow import keras

from tensorflow.keras import layers

from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Concatenate , Add, Dot, Activation, Lambda
from tensorflow.keras.models import Model

from tensorflow.image import flip_up_down, flip_left_right, rot90
from tensorflow.linalg import normalize

from tensorflow.keras.preprocessing.image import ImageDataGenerator

import math

import matplotlib.pyplot as plt
import sys 

class SortedConv2D(tf.keras.layers.Layer):
    def __init__(self, filters, padding = 'VALID', strides = (1, 1), activation=None, use_bias = True):
        super(SortedConv2D, self).__init__()
        self.filters = filters
        self.kernel_size = (3,3)
        self.activation = activation
        self.padding = padding
        self.kernel_initializer = tf.keras.initializers.GlorotNormal(seed=5)
        self.param_initializer =tf.keras.initializers.GlorotNormal(seed=5)

        self.bias_initializer = tf.initializers.Zeros()
        self.strides = strides
        self.use_bias = use_bias

        self.w_a = None     # AntiSymetric kernel weights
        self.w_s = None     # Symetric kernel weights

        self.sym_param_a = None
        self.sym_param_b= None
        self.sym_param_c = None

        self.bias = None

        self.scale = None

    def get_config(self):
        config = super().get_config()
        config.update({
            "filters": self.filters,
            "padding": self.padding,
            "strides": self.strides,
            "activation": self.activation,
            "use_bias": self.use_bias,
        })
        return config

    def get_weights(self):
        return tf.stack([self.w_a, self.w_s])

    def build(self, input_shape):
        *_, n_channels = input_shape

        #####  BUILD Fa   ##### 
        t = tf.repeat([tf.expand_dims(tf.linspace(math.pi/self.filters, math.pi, self.filters, axis=0),axis=0)], n_channels, axis=1)

        a = -tf.math.sqrt(8.0)*tf.math.cos(t - 9*math.pi/4)
        b = -2*tf.math.sin(t)
        c = -tf.math.sqrt(8.0)*tf.math.sin(t - 9*math.pi/4)
        d = -2*tf.math.cos(t)
        
        self.scale = tf.Variable(1.0, trainable=True)
        self.w_a   = tf.Variable(initial_value = tf.stack([tf.concat([a,b,c], axis=0) , 
                                 tf.concat( [d,tf.zeros([1, n_channels, self.filters]), -d], axis=0),
                                 tf.concat( [-c, -b, -a], axis=0)]), trainable=True)

        #####  BUILD Fs   ##### 
        self.sym_param_a = tf.Variable(
            initial_value=self.param_initializer(shape=(1,
                                                        n_channels,
                                                        self.filters),
                                 dtype='float32'), trainable=True)
        self.sym_param_b = tf.Variable(
            initial_value=self.param_initializer(shape=(1,
                                                        n_channels,
                                                        self.filters),
                                 dtype='float32'), trainable=True)
        self.sym_param_c = tf.Variable(
            initial_value=self.param_initializer(shape=(1,
                                                        n_channels,
                                                        self.filters),
                                 dtype='float32'), trainable=True)

        if self.use_bias:
            self.bias = tf.Variable(
                initial_value=self.bias_initializer(shape=(self.filters,), 
                                                    dtype='float32'),
                trainable=True)


    def call(self, inputs, training=None):

        x_a =  tf.nn.conv2d(inputs, filters= self.w_a , strides=self.strides, 
                          padding=self.padding)

        x_a = tf.math.scalar_mul(self.scale, x_a)

        self.w_s  = tf.stack([tf.concat([self.sym_param_a, self.sym_param_b, self.sym_param_a], axis=0), 
                              tf.concat([self.sym_param_b, self.sym_param_c, self.sym_param_b], axis=0),
                              tf.concat([self.sym_param_a, self.sym_param_b, self.sym_param_a], axis=0)])

        x_s =  tf.nn.conv2d(inputs, filters=self.w_s, strides=self.strides, 
                          padding=self.padding)
        if self.use_bias:
            x_s = x_s + self.bias

        return x_a+x_s
        
