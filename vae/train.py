from __future__ import absolute_import, division, print_function

import argparse
import sys
import os
from typing import Optional, Union

import tensorflow as tf
import numpy as np
from tensorflow.keras.optimizers import Adam, RMSprop, Adagrad
from tensorflow.keras.layers import Lambda
from tensorflow.keras.models import Model
from tensorflow.keras import Input
import tensorflow.keras.backend as KB
from .model import Encoder, Decoder
from .loss import BinomialVLB, NormalVLB
from .util import get_params

OPTIMIZERS = {
    "rmsprop": RMSprop,
    "adam": Adam,
    "adagrad": Adagrad
}

class Trainer(object):
    def __init__(self):
        self.params = get_params("vae/params.json")

        self.input_dims = self.params['input_dims']
        self.latent_dims = self.params['latent_dims']
        self.batch_size = self.params['batch_size']

        self.x = Input(batch_size=(self.batch_size, *self.input_dims))

        self.encoder = Encoder(self.params['encoder'])(x)
        self.get_loc = Lambda(lambda t: t[:, :latent_dims])(self.encoder)
        self.get_log_var = Lambda(lambda t: t[:, latent_dims:])(self.encoder)
        self.sample_z = Lambda(self._sampling)([self.get_loc, self.get_var])
        self.decoder = Decoder(self.params['decoder'])(self.sample_z)

        self.model = Model(inputs=[self.x], outputs=[self.decoder, self.get_loc, self.get_log_var])

        self.loss = getattr(self, self.params['loss'])

        self.learning_rate = self.params['learning_rate']
        self.optimizer = OPTIMIZERS[self.params['optimizer']](learning_rate=self.learning_rate)

        self.epochs = self.params['epochs']

        self.model.compile(self.optimizer, lambda *args, **kwargs: loss(*args, **kwargs))

    @property
    def normal_vlb(self):
        return NormalVLB

    @property
    def binomial_vlb(self):
        return BinomialVLB
    
    def _sampling(self, args):
        loc, log_var = args
        epsilon = KB.random_normal(loc.shape)
        z = loc + KB.exp( 0.5 * log_var ) * epsilon
        return z

    def train(self, x_train, y_train, x_test, y_test):
        hist = self.model.fit(x=x_train, y=x_train,
                              shuffle=True,
                              epochs=self.epochs,
                              batch_size=self.batch_size,
                              validation_data=(x_test, x_test),
                              verbose=2)
