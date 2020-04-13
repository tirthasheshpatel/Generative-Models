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
from vae.model import Encoder, Decoder
from vae.loss import BinomialVLB, NormalVLB
from vae.util import get_params

OPTIMIZERS = {
    "RMSprop": RMSprop,
    "Adam": Adam,
    "Adagrad": Adagrad
}

class Trainer(object):
    def __init__(self):
        self.params = get_params("vae/params.json")

        self.input_dims = self.params['input_dims']
        self.latent_dims = self.params['latent_dims']
        self.batch_size = self.params['batch_size']

        self.x = Input(batch_shape=(self.batch_size, *self.input_dims))

        self.encoder = Encoder(**self.params['encoder'])
        self.encoded = self.encoder(self.x)
        self.get_loc = Lambda(lambda t: t[:, :self.latent_dims])
        self.get_log_var = Lambda(lambda t: t[:, self.latent_dims:])
        self.loc = self.get_loc(self.encoded)
        self.log_var = self.get_log_var(self.encoded)
        self.sample_z = Lambda(self._sampling)
        self.z = self.sample_z([self.loc, self.log_var])
        self.decoder = Decoder(**self.params['decoder'])
        self.decoded = self.decoder(self.z)

        self.model = Model(self.x, self.decoded)

        self.loss = getattr(self, self.params['loss'])(self.x, self.decoded, self.loc, self.log_var)

        self.learning_rate = self.params['learning_rate']
        self.optimizer = OPTIMIZERS[self.params['optimizer']](learning_rate=self.learning_rate)

        self.epochs = self.params['epochs']

        self.model.compile(self.optimizer, lambda *args, **kwargs: self.loss)

    @property
    def normal_vlb(self):
        return NormalVLB.unsigned_call

    @property
    def binomial_vlb(self):
        return BinomialVLB.unsigned_call
    
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
                              verbose=1)
