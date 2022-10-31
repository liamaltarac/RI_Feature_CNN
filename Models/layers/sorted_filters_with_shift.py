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

class SortedConv2DWithShift(tf.keras.layers.Layer):
    def __init__(self, filters, padding = 'VALID', strides = (1, 1), activation=None, use_bias = True, patch_size=4):
        super(SortedConv2DWithShift, self).__init__()
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

    def get_config(self):
        config = super().get_config()
        config.update({
            "filters": self.filters,
            "padding": self.padding,
            "strides": self.strides,
            "activation": self.activation,
            "use_bias": self.use_bias,
            "patch_size": self.patch_size,
        })
        return config

    def get_weights(self):
        return tf.stack([self.w_a, self.w_s])

    def build(self, input_shape):
        *_, n_channels = input_shape

        self.batch_size = input_shape[0]
        self.image_h = input_shape[1]
        self.image_w = input_shape[2]

        self.channel_size = self.filters

        self.num_patches = int((self.image_h * self.image_w ) / (self.patch_size * self.patch_size))


        #####  BUILD Fa   ##### 
        t = tf.repeat([tf.expand_dims(tf.linspace(0.0, math.pi, self.filters, axis=0),axis=0)], n_channels, axis=1)
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

        self.scale_s = tf.Variable(initial_value=self.sym_initializer(shape=(1,)), trainable=True, name="scale_sym")  #tf.Variable(initial_value=tf.math.abs(tf.reduce_mean(self.sym_param_a)) * 2.0, trainable=True) 
        self.scale_a = tf.Variable(initial_value=self.asym_initializer(shape=(1,)), trainable=True, name="scale_asym")  #tf.Variable(initial_value=tf.math.abs(tf.reduce_mean(self.sym_param_a)) * 2.0, trainable=True) 
            
        #self.scale_a = tf.Variable(self.gain_initializer(shape=(1,)), trainable=True, name="scale_asym")  #tf.Variable(initial_value=tf.math.abs(tf.reduce_mean(self.sym_param_a)) * 2.0, trainable=True) 
     

        #self.gain = tf.Variable(initial_value=self.gain_initializer(shape=(self.filters,)), trainable=True, name="gain")


    def call(self, inputs, training=None):

        x_a =   tf.nn.conv2d(inputs, filters=tf.math.multiply(self.scale_a,self.w_a) , strides=self.strides, 
                          padding=self.padding)
        
        w_s  = tf.stack([tf.concat([self.sym_param_a, self.sym_param_b, self.sym_param_a], axis=0), 
                              tf.concat([self.sym_param_b, self.sym_param_c, self.sym_param_b], axis=0),
                              tf.concat([self.sym_param_a, self.sym_param_b, self.sym_param_a], axis=0)])

        '''x =  tf.nn.conv2d(inputs, filters=  tf.math.scalar_mul(self.scale , tf.math.add(self.w_a, w_s))  , strides=self.strides, 
                          padding=self.padding)'''

        x_s =   tf.nn.conv2d(inputs, filters=tf.math.multiply(self.scale_s, w_s) , strides=self.strides, 
                          padding=self.padding)

        x =  tf.math.add(x_a , x_s)
        if self.use_bias:
            #x_s = x_s + self.bias
            x = x+self.bias

        x = self.activation(x)
        
        map = tf.math.argmax(x_a, axis=-1)
        map  = tf.cast(map, tf.float32)
        map = tf.nn.avg_pool2d((tf.expand_dims(map, axis=-1)), ksize=[self.patch_size, self.patch_size] , strides=[self.patch_size, self.patch_size], padding='VALID')
        map  = tf.cast(map, tf.int32)


        x = tf.reshape(x, [-1, self.image_h*self.image_w, self.channel_size ])
        m = tf.reshape(map, [-1, map.shape[1]*map.shape[2]])

        #x_s = self.activation(x_s)

        # Build map of dominant orientataion
        #shifted, map = self.shift_to_max(x, x_a)

        out = tf.vectorized_map(self.n_roll, (x, -m), fallback_to_while_loop=False, warn=True)
        print('out shape', x.shape)
        out = tf.reshape(out, [-1, self.image_h, self.image_w, self.channel_size ])


        return  x, m  #self.activation(tf.math.add(x_a, x_s)) #, self.map
        

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

    def roll(self, x, map):
        def roll_fn(args):

            x, map = args
            print(tf.executing_eagerly())
            out = tf.zeros([x.shape[0], x.shape[1], x.shape[2], x.shape[3]])

            print("000")
            print
            for i in range(x.shape[1]):
                for j in range(x.shape[2]):
                    print("pppp")
                    out[i,j] = tf.roll(x[i,j], shift=map[i,j], axis=-1)
            return out
        print("doing")
        return tf.map_fn(roll_fn, (x, map))

        #shifted = tf.TensorArray(tf.float32, size=map.shape[-1], dynamic_size=False, infer_shape=True)
        '''print('here', map.shape)
        print('here', x.shape)

        for i in range(map.shape[-1]):

            tf.autograph.experimental.set_loop_options(
                shape_invariants=[(x, tf.TensorShape([None, None, None, None]))]
            )
            tmp = tf.roll(x[i], shift=map[i], axis=-1)
            #print(x[i], tmp)
            shifted = shifted.write(i, tmp)  # shape is (n_atoms, n_timesteps, n_atoms, n_atoms)
        out = shifted.stack()
        print("done")'''
        print("111")
        return x

    def n_roll(self,  arg):
        x, map = arg
        shifted = tf.zeros([0, self.channel_size]) #tf.TensorArray(tf.float32, size=40, dynamic_size=False)
        for i in range(self.num_patches):
            print(i)
            k = map[i] % self.channel_size 
            shifted = tf.concat([shifted, tf.concat(
                                                [tf.slice(x, [i*self.patch_size*self.patch_size, self.channel_size-k], [self.patch_size*self.patch_size, k]),
                                                 tf.slice(x, [i*self.patch_size*self.patch_size, 0], [self.patch_size*self.patch_size, self.channel_size-k])], 
                                           1)],
                                 0)
        return shifted



