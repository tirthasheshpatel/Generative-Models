import tensorflow as tf
from tensorflow.keras.layers import Lambda, Reshape, Conv2D, Conv2DTranspose, Dense, InputLayer, Flatten
from tensorflow.keras.models import Model, Sequential
import tensorflow.keras.backend as KB

KB.set_image_data_format('channels_last')
KB.set_floatx('float32')

def Encoder(input_dims, channels, kernel_widths, strides,
            hidden_activation, output_activation,
            latent_dims, name="encoder"):
    input_layer = InputLayer(input_dims)
    encoder_layers = []
    for channel, kernel_width, stride in zip(channels, kernel_widths, strides):
        layer = Conv2D(channel, kernel_width, stride,
                       activation=hidden_activation)
        encoder_layers.append(layer)
    encoder_layers.append(Flatten())
    # self._output_layer = Dense(latent_dims + latent_dims, output_activation)
    output_layer = Dense(latent_dims + latent_dims)
    model = Sequential([input_layer, *encoder_layers, output_layer])
    return model


def Decoder(initial_dense_dims, initial_activation,
            starting_target_shape,
            channels, kernel_widths,
            strides, hidden_activation,
            latent_dims, name="decoder"):
    decoder_layers = []
    input_layer = InputLayer([latent_dims])
    decoder_layers.append( Dense(initial_dense_dims, initial_activation) )
    decoder_layers.append( Reshape(starting_target_shape) )
    for channel, kernel_width, stride in zip(channels, kernel_widths, strides):
        layer = Conv2DTranspose(channel, kernel_width, stride,
                                padding='SAME',
                                activation=hidden_activation)
        decoder_layers.append(layer)
    output_layer = Conv2DTranspose(1, 3, 1, padding="SAME", activation='sigmoid')
    model = Sequential([input_layer, *decoder_layers, output_layer])
    return model
