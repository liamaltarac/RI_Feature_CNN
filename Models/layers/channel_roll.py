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

class ChannelRoll(tf.keras.layers.Layer):
    def __init__(self):
        super(ChannelRoll, self).__init__(trainable= False)

    def call(self, inputs, training=None):

        x = inputs[0]
        map = inputs[1]

        map  = tf.cast(map, tf.int64)

        index = tf.squeeze(tf.sequence_mask(map, x.shape[-1]), axis=-2)
        index = tf.math.logical_not(index)
        out_left = tf.ragged.boolean_mask(x, index)

        index = tf.math.logical_not(index)
        out_right = tf.ragged.boolean_mask(x, index)

        out_shifted = tf.concat([out_left, out_right], axis=-1)

        return out_shifted.to_tensor(shape=x.shape) #self.activation(tf.math.add(x_a, x_s)) #, self.map
        
