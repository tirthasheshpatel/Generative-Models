import os

import keras

IMG_SIZE = int(os.getenv('IMG_SIZE'))

def GANDiscriminator():
    model = keras.applications.InceptionV3(include_top=False,
                                           input_shape=(IMG_SIZE, IMG_SIZE, 3),
                                           weights='imagenet')
    new_output = keras.layers.GlobalAveragePooling2D()(model.output)
    new_output = keras.layers.Dense(1, activation='sigmoid')(new_output)
    # model = keras.engine.training.Model(model.inputs, new_output)

    for layer in model.layers:
        layer.trainable = True
        if isinstance(layer, keras.layers.BatchNormalization):
            layer.momentum = 0.9

    for layer in model.layers[:-50]:
        if not isinstance(layer, keras.layers.BatchNormalization):
            layer.trainable = False

    return model
