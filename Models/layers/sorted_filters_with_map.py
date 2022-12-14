from re import X
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
from tensorflow.compat.v1 import extract_image_patches

import math

import matplotlib.pyplot as plt
import sys 

class SortedConv2DWithMap(tf.keras.layers.Layer):
    def __init__(self, filters, layer_num, padding = 'VALID', strides = (1, 1), activation=None, use_bias = True, patch_size=4):
        super(SortedConv2DWithMap, self).__init__()
        self.filters = filters
        self.kernel_size = (3,3)
        self.activation = activation
        self.padding = padding
        self.kernel_initializer = tf.keras.initializers.GlorotNormal(seed=5)
        self.param_initializer =tf.keras.initializers.GlorotNormal(seed=5)
        
        self.sym_initializer = tf.keras.initializers.RandomUniform(
                                    minval=-2, maxval=0.1, seed=5
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
        self.patch_size = patch_size

        self.batch_size = None
        self.image_h = None
        self.image_w = None

        self.channel_size = None

        self.num_patches = None
        self.layer_num = layer_num
    def get_config(self):
        config = super().get_config()
        config.update({
            "filters": self.filters,
            "padding": self.padding,
            "strides": self.strides,
            "activation": self.activation,
            "use_bias": self.use_bias,
            "patch_size": self.patch_size,
            "layer_num": self.layer_num
        })
        return config

    def get_weights(self):
        return tf.stack([self.w_a, self.w_s])

    def build(self, input_shape):
        *_, n_channels = input_shape

        '''self.batch_size = input_shape[0]
        self.image_h = input_shape[1]
        self.image_w = input_shape[2]'''

        self.channel_size = self.filters


        #print( self.filters)

        #####  BUILD Fa   ##### 
        t = tf.random.normal([1, n_channels, self.filters], mean=tf.linspace(0.0, 2* math.pi,  self.filters, axis=0), stddev=self.layer_num/(math.pi**2))
        #t = tf.repeat([tf.expand_dims(tf.linspace(0.0, math.pi, self.filters, axis=0),axis=0)], n_channels, axis=1)
        #print(t)
        #t = tf.repeat([tf.expand_dims(tf.repeat(tf.linspace(0.0, 2.0*math.pi, 8, axis=0), self.filters//8, axis=0),axis=0)], n_channels, axis=1)

        a = -tf.math.sqrt(8.0)*tf.math.cos(t - 9*math.pi/4)
        b = -2*tf.math.sin(t)
        c = -tf.math.sqrt(8.0)*tf.math.sin(t - 9*math.pi/4)
        d = -2*tf.math.cos(t)
        
        #self.scale = tf.Variable(0.0, trainable=True)                  
        #tf.Variable(0.01, trainable=True)
        self.w_a =  tf.stack([tf.concat( [a,b,c], axis=0) , 
                              tf.concat( [d,tf.zeros([1, n_channels, self.filters]), -d], axis=0),
                              tf.concat( [-c, -b, -a], axis=0)])
        #####  BUILD Fs   ##### 
        self.sym_param_a = tf.Variable(
            initial_value=self.sym_initializer(shape=(1,
                                                        n_channels,
                                                        self.filters),
                                 dtype='float32'), trainable=True , name="sym_param_a" )
        self.sym_param_b = tf.Variable(
            initial_value=self.sym_initializer(shape=(1,
                                                        n_channels,
                                                        self.filters),
                                 dtype='float32'), trainable=True , name="sym_param_b" )
        self.sym_param_c = tf.Variable(
            initial_value=self.sym_initializer(shape=(1,
                                                        n_channels,
                                                        self.filters),
                                 dtype='float32'), trainable=True , name="sym_param_c" )

        '''self.w_s  = tf.stack([tf.concat([self.sym_param_a, self.sym_param_b, self.sym_param_a], axis=0), 
                                          tf.concat([self.sym_param_b, self.sym_param_c, self.sym_param_b], axis=0),
                                          tf.concat([self.sym_param_a, self.sym_param_b, self.sym_param_a], axis=0)])'''


        if self.use_bias:
            self.bias = tf.Variable(
                initial_value=self.bias_initializer(shape=(self.filters,), 
                                                    dtype='float32'),
                trainable=True, name="bias" )

        #self.scale_s = tf.Variable(initial_value=self.sym_initializer(shape=(1, 1, 1, self.filters)), trainable=True, name="scale_sym")  #tf.Variable(initial_value=tf.math.abs(tf.reduce_mean(self.sym_param_a)) * 2.0, trainable=True) 
        #self.scale_a = tf.Variable(initial_value=self.asym_initializer(shape=(1,1,1, 1)), trainable=True, name="scale_asym")  #tf.Variable(initial_value=tf.math.abs(tf.reduce_mean(self.sym_param_a)) * 2.0, trainable=True) 
            
        self.scale_a = tf.Variable(self.gain_initializer(shape=(1,)), trainable=True, name="scale_asym")  #tf.Variable(initial_value=tf.math.abs(tf.reduce_mean(self.sym_param_a)) * 2.0, trainable=True) 
     

        #self.gain = tf.Variable(initial_value=self.gain_initializer(shape=(self.filters,)), trainable=True, name="gain")


    def call(self, inputs, training=None):


        x_a =   tf.nn.conv2d(inputs, filters=   self.scale_a * self.w_a  , strides=self.strides, 
                          padding=self.padding)
        
        w_s  = tf.stack([tf.concat([self.sym_param_a, self.sym_param_b, self.sym_param_a], axis=0), 
                              tf.concat([self.sym_param_b, self.sym_param_c, self.sym_param_b], axis=0),
                              tf.concat([self.sym_param_a, self.sym_param_b, self.sym_param_a], axis=0)])

        '''x =  tf.nn.conv2d(inputs, filters=  tf.math.scalar_mul(self.scale , tf.math.add(self.w_a, w_s))  , strides=self.strides, 
                          padding=self.padding)'''

        x_s =  tf.nn.conv2d(inputs, filters=   w_s , strides=self.strides, 
                          padding=self.padding)

        x =  tf.math.add( x_a, x_s)
        if self.use_bias:
            #x_s = x_s + self.bias
            x = x+self.bias

        x = self.activation(x)
        
        map = tf.math.argmax(x_a, axis=-1)
        #map = tf.expand_dims(map, axis=-1)
        map  = tf.cast(map, tf.float32)
        map = tf.nn.avg_pool2d((tf.expand_dims(map, axis=-1)), ksize=[self.patch_size, self.patch_size] , strides=[self.patch_size, self.patch_size], padding='SAME')

        map = tf.repeat(map, repeats = self.patch_size, axis=1)
        map = tf.repeat(map, repeats = self.patch_size, axis=2)

        '''map  = tf.cast(map, tf.int64)

        index = tf.squeeze(tf.sequence_mask(map, self.filters), axis=-2)
        index = tf.math.logical_not(index)
        out_left = tf.ragged.boolean_mask(x, index)

        index = tf.math.logical_not(index)
        out_right = tf.ragged.boolean_mask(x, index)

        out_shifted = tf.concat([out_left, out_right], axis=-1)''''''
        
        #print('xs : ', x_shifted.shape)

        #x_shifted = tf.reshape(x_shifted, [-1, x.shape[1], x.shape[2], x.shape[3]])

        #x_s = self.activation(x_s)

        # Build map of dominant orientataion
        #shifted, map = self.shift_to_max(x, x_a)

        '''
        '''print('out shape', x.shape)
        out = tf.reshape(out, [-1, self.image_h, self.image_w, self.channel_size ])'''


        return  x, map # out_shifted.to_tensor(shape=[x.shape[0], x.shape[1], x.shape[2], x.shape[3]]) #self.activation(tf.math.add(x_a, x_s)) #, self.map
        

    def get_scale(self):
        return self.scale

    def get_asym_filter(self, channel, filter):
        return self.w_a[:,:,channel,filter]
    
    def get_sym_filter(self, channel, filter):
        ws = tf.stack([tf.concat([self.sym_param_a, self.sym_param_b, self.sym_param_a], axis=0), 
                       tf.concat([self.sym_param_b, self.sym_param_c, self.sym_param_b], axis=0),
                       tf.concat([self.sym_param_a, self.sym_param_b, self.sym_param_a], axis=0)])
        return ws[:,:,channel,filter]

    def get_map(self):
        return self.map



