import os

import keras

LATENT_DIMS = int(os.getenv('LATENT_DIMS'))
IMG_SIZE = int(os.getenv('IMG_SIZE'))

def GANGenerator():
    model = keras.models.Sequential(name='generator')
    model.add(keras.layers.InputLayer(input_shape=[LATENT_DIMS]))
    model.add(keras.layers.Reshape(target_shape=[1, 1, LATENT_DIMS]))
    model.add(keras.layers.Conv2DTranspose(filters=1024,
                                           kernel_size=8,
                                           strides=1,
                                           padding='VALID',
                                           data_format='channels_last',
                                           activation='relu'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Conv2DTranspose(filters=512,
                                           kernel_size=3,
                                           strides=2,
                                           padding='SAME',
                                           data_format='channels_last',
                                           activation='relu'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Conv2DTranspose(filters=256,
                                           kernel_size=3,
                                           strides=2,
                                           padding='SAME',
                                           data_format='channels_last',
                                           activation='relu'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Conv2DTranspose(filters=128,
                                           kernel_size=3,
                                           strides=2,
                                           padding='SAME',
                                           data_format='channels_last',
                                           activation='relu'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Conv2DTranspose(filters=64,
                                           kernel_size=3,
                                           strides=2,
                                           padding='SAME',
                                           data_format='channels_last',
                                           activation='relu'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Conv2DTranspose(filters=3,
                                           kernel_size=3,
                                           strides=2,
                                           padding='SAME',
                                           data_format='channels_last',
                                           activation='relu'))
    return model