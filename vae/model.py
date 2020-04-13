import tensorflow as tf
from tensorflow.keras.layers import Lambda, Reshape, Conv2D, Conv2DTranspose, Dense, InputLayer, Flatten
from tensorflow.keras.models import Model, Sequential
import tensorflow.keras.backend as KB

KB.set_image_data_format('channels_last')
KB.set_floatx('float32')

class Encoder(object):
    def __init__(self, input_dims, channels, kernel_widths, strides,
                 hidden_activation, output_activation,
                 latent_dims, name="encoder"):
        self.name = name
        self._input_layer = InputLayer(input_dims)
        self._latent_dims = latent_dims
        self._encoder_layers = []
        for channel, kernel_width, stride in zip(channels, kernel_widths, strides):
            _layer = Conv2D(channel, kernel_width, stride,
                            activation=hidden_activation)
            self._encoder_layers.append(_layer)
        self._encoder_layers.append(Flatten())
        self._output_layer = Dense(latent_dims + latent_dims, output_activation)
        self._model = Sequential([self._input_layer, *self._encoder_layers, self._output_layer])

    def __call__(self, inputs):
        encoded = self._model(inputs)
        return encoded

class Decoder(object):
    def __init__(self, initial_dense_dims, initial_activation,
                 starting_target_shape,
                 channels, kernel_widths,
                 strides, hidden_activation,
                 latent_dims, name="decoder"):
        self.name = name
        self._latent_dims = latent_dims
        self._decoder_layers = []
        self._input_layer = InputLayer([latent_dims])
        self._decoder_layers.append( Dense(initial_dense_dims, initial_activation) )
        self._decoder_layers.append( Reshape(starting_target_shape) )
        for channel, kernel_width, stride in zip(channels, kernel_widths, strides):
            _layer = Conv2DTranspose(channel, kernel_width, stride,
                                     padding='SAME',
                                     activation=hidden_activation)
            self._decoder_layers.append(_layer)
        self._output_layer = Conv2DTranspose(1, 3, 1, padding="SAME")
        self._model = Sequential([self._input_layer, *self._decoder_layers, self.output_layer])

    def __call__(self, inputs):
        decoded = self._model(inputs)
        return decoded
