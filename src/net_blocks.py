import keras
import numpy as np
from keras.engine import InputSpec
from keras.engine.topology import Layer
import tensorflow as tf
class ReflectionPadding3D(Layer):
    def __init__(self, padding=(1, 1 ,1), **kwargs):
        self.padding = tuple(padding)
        self.input_spec = [InputSpec(ndim=5)]
        super(ReflectionPadding3D, self).__init__(**kwargs)

    def compute_output_shape(self, s):
        """ If you are using "channels_last" configuration"""
        return (s[0], s[1] + 2 * self.padding[0], s[2] + 2 * self.padding[1], s[3]+2*self.padding[2],s[4])

    def call(self, x, mask=None):
        w_pad,h_pad,z_pad = self.padding
        return tf.pad(x, [[0,0], [h_pad,h_pad], [w_pad,w_pad] , [z_pad,z_pad], [0,0] ], 'REFLECT')

def conv3D_block(n_filters,kernel,input,stride=1,name=None,concat = None,dropout=0,bias_reg=None):
    if concat != None:
        input= keras.layers.Concatenate()([input,concat])
    x = ReflectionPadding3D((int(np.floor(kernel[0]/2)), int(np.floor(kernel[1]/2)), int(np.floor(kernel[2]/2))))(input)
    x = keras.layers.Conv3D(n_filters, kernel, padding='valid',strides=stride, name=name, activation=None,bias_regularizer=bias_reg)(x)
    x = keras.layers.Activation('relu')(x)
    if dropout!=0:
       x = keras.layers.Dropout(dropout)(x)

    return x

def squeeze(x):
    import keras

    return keras.backend.squeeze(x,axis=-1)

def expand(x):
    import keras
    return keras.backend.expand_dims(x,axis=-1)

