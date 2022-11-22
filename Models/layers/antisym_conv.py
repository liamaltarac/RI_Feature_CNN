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

class AntiSymConv2D(tf.keras.layers.Layer):
    def __init__(self, filters, padding = 'VALID', strides = (1, 1), activation=None, use_bias = True):
        super(AntiSymConv2D, self).__init__()
        self.filters = filters
        self.kernel_size = (3,3)
        self.activation = activation
        self.padding = padding
        self.kernel_initializer = tf.keras.initializers.GlorotNormal(seed=5)
        self.param_initializer =tf.keras.initializers.GlorotNormal(seed=5)
        
        self.sym_initializer = tf.keras.initializers.GlorotNormal(
                                    seed=5
                                )
        self.asym_initializer = tf.keras.initializers.RandomUniform(
                                    minval=-0.5, maxval=0.5, seed=5
                                )

        self.gain_initializer = tf.keras.initializers.RandomUniform(
                                    minval=-0.1, maxval=2, seed=5
                                )
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

        self.map = None
        self.n_channels = None
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
        
        
        w_a =  tf.Variable(initial_value=tf.stack([tf.concat( [self.asym_param_a,self.asym_param_b,self.asym_param_c], axis=0) , 
                              tf.concat( [self.asym_param_d,tf.zeros([1, self.n_channels, self.filters]), -self.asym_param_d], axis=0),
                              tf.concat( [-self.asym_param_c, -self.asym_param_b, -self.asym_param_a], axis=0)]), trainable=True , name="asym_" )

        return w_a, self.bias
    

    def build(self, input_shape):
        *_, self.n_channels = input_shape

        #####  BUILD Fa   ##### 
        #t = tf.random.normal([1, n_channels, self.filters], mean=tf.linspace(0.0, 2* math.pi,  self.filters, axis=0), stddev=100)

        #self.scale = tf.Variable(0.0, trainable=True)                  
        #tf.Variable(0.01, trainable=True)
        '''self.w_a =  tf.Variable(initial_value=tf.stack([tf.concat( [a,b,c], axis=0) , 
                              tf.concat( [d,tf.zeros([1, n_channels, self.filters]), -d], axis=0),
                              tf.concat( [-c, -b, -a], axis=0)]), trainable=True , name="asym_" ) * tf.math.reduce_mean(self.kernel_initializer(shape=(1, n_channels, self.filters), dtype=tf.float32))''' 
        #####  BUILD Fs   ##### 
        self.asym_param_a = tf.Variable(
            initial_value=self.sym_initializer(shape=(1,
                                                        self.n_channels,
                                                        self.filters),
                                 dtype='float32'), trainable=True , name="asym_param_a" )
        self.asym_param_b = tf.Variable(
            initial_value=self.sym_initializer(shape=(1,
                                                        self.n_channels,
                                                        self.filters),
                                 dtype='float32'), trainable=True , name="asym_param_b" )
        self.asym_param_c = tf.Variable(
            initial_value=self.sym_initializer(shape=(1,
                                                        self.n_channels,
                                                        self.filters),
                                 dtype='float32'), trainable=True , name="asym_param_c" )

        self.asym_param_d = tf.Variable(
            initial_value=self.sym_initializer(shape=(1,
                                                        self.n_channels,
                                                        self.filters),
                                 dtype='float32'), trainable=True , name="asym_param_d" )  

        '''self.w_s  = tf.stack([tf.concat([self.sym_param_a, self.sym_param_b, self.sym_param_a], axis=0), 
                                          tf.concat([self.sym_param_b, self.sym_param_c, self.sym_param_b], axis=0),
                                          tf.concat([self.sym_param_a, self.sym_param_b, self.sym_param_a], axis=0)])'''


        if self.use_bias:
            self.bias = tf.Variable(
                initial_value=self.bias_initializer(shape=(self.filters,), 
                                                    dtype='float32'),
                trainable=True, name="bias" )

        #self.scale_a = tf.Variable(initial_value=self.asym_initializer(shape=(1,)), trainable=True, name="scale_asym")  #tf.Variable(initial_value=tf.math.abs(tf.reduce_mean(self.sym_param_a)) * 2.0, trainable=True) 
            
        #self.scale_a = tf.Variable(self.gain_initializer(shape=(1,)), trainable=True, name="scale_asym")  #tf.Variable(initial_value=tf.math.abs(tf.reduce_mean(self.sym_param_a)) * 2.0, trainable=True) 
     

        #self.gain = tf.Variable(initial_value=self.gain_initializer(shape=(self.filters,)), trainable=True, name="gain")


    def call(self, inputs, training=None):


        
        w_a =  tf.stack([tf.concat( [self.asym_param_a,self.asym_param_b,self.asym_param_c], axis=0) , 
                              tf.concat( [self.asym_param_d,tf.zeros([1, self.n_channels, self.filters]), -self.asym_param_d], axis=0),
                              tf.concat( [-self.asym_param_c, -self.asym_param_b, -self.asym_param_a], axis=0)])

        x =   tf.nn.conv2d(inputs, w_a , strides=self.strides, 
                          padding=self.padding)

        '''x =  tf.nn.conv2d(inputs, filters=  tf.math.scalar_mul(self.scale , tf.math.add(self.w_a, w_s))  , strides=self.strides, 
                          padding=self.padding)'''

        if self.use_bias:
            #x_s = x_s + self.bias
            x = x+self.bias
        #self.map = tf.math.argmax(x_a, axis=-1)

        #x_s = self.activation(x_s)

        if self.activation :
            return  self.activation(x)  #self.activation(tf.math.add(x_a, x_s)) #, self.map
        else:
            return x
        
    def get_scale(self):
        return self.scale


    def get_map(self):
        return self.map