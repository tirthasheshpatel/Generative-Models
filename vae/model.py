import warnings
import sys
import os
import absl

import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Lambda, Reshape, Conv2D, Conv2DTranspose, UpSampling2D, Dense, Input, InputLayer, Flatten, GaussianDropout, Layer
from tensorflow.keras.models import Model, Sequential
import tensorflow.keras.backend as KB

KB.set_image_data_format('channels_last')
KB.set_floatx('float32')

class Encoder(Layer):
    def __init__(self, channels, kernel_widths, strides,
                 hidden_activation, output_activation,
                 latent_dims, name="encoder"):
        super(Encoder, self).__init__(name=name)
        self._latent_dims = latent_dims
        self._encoder_layers = []
        for channel, kernel_width, stride in zip(channels, kernel_widths, strides):
            _layer = Conv2D(channel, kernel_width, stride,
                            activation=hidden_activation)
            self._encoder_layers.append(_layer)
        self._encoder_layers.append(Flatten())
        self._output_layer = Dense(latent_dims + latent_dims, output_activation)

    def call(self, inputs):
        return self.unsigned_call(inputs)

    def unsigned_call(self, inputs):
        X = self._encoder_layers[0](inputs)
        for layer in self._encoder_layers[1:]:
            X = layer(X)
        encoded = self._output_layer(X)
        return encoded

class Decoder(Layer):
    def __init__(self, initial_dense_dims, initial_activation,
                 starting_target_shape,
                 channels, kernel_widths,
                 strides, hidden_activation,
                 latent_dims, name="decoder"):
        super(Decoder, self).__init__(name=name)
        self._latent_dims = latent_dims
        self._decoder_layers = []
        self._decoder_layers.append( Dense(initial_dense_dims, initial_activation) )
        self.decoder_layers.append( Reshape(starting_target_shape) )
        for channel, kernel_width, stride in zip(channels, kernel_widths, strides):
            _layer = Conv2DTranspose(channel, kernel_width, stride,
                                     padding='SAME',
                                     activation=hidden_activation)
            self.decoder_layers.append(_layer)
        self._output_layer = Conv2DTranspose(1, 3, 1, padding="SAME")

    def call(self, inputs):
        return self.unsigned_call(inputs)

    def unsigned_call(self, inputs):
        X = self.decoder_layers[0](inputs)
        for layer in self.decoder_layers[1:]:
            X = layer(X)
        decoded = self._output_layer(X)
        return decoded
