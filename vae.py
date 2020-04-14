import tensorflow as tf
import keras
import numpy as np
import matplotlib.pyplot as plt

from keras.layers import Input, Dense, Lambda, InputLayer, concatenate, Dropout
from keras.models import Model, Sequential
from keras import backend as K
from keras import metrics
from keras.datasets import mnist
from keras.utils import np_utils

# Start tf session so we can run code.
sess = tf.InteractiveSession()
# Connect keras to the created session.
K.set_session(sess)


def vlb_binomial(x, x_decoded_mean, t_mean, t_log_var):
    """Returns the value of negative Variational Lower Bound
    
    The inputs are tf.Tensor
        x: (batch_size x number_of_pixels) matrix with one image per row with zeros and ones
        x_decoded_mean: (batch_size x number_of_pixels) mean of the distribution p(x | t), real numbers from 0 to 1
        t_mean: (batch_size x latent_dim) mean vector of the (normal) distribution q(t | x)
        t_log_var: (batch_size x latent_dim) logarithm of the variance vector of the (normal) distribution q(t | x)
    
    Returns:
        A tf.Tensor with one element (averaged across the batch), VLB
    """
    vlb = tf.reduce_mean(
        tf.reduce_sum(
            x * tf.log(x_decoded_mean + 1e-19)
            + (1 - x) * tf.log(1 - x_decoded_mean + 1e-19),
            axis=1,
        )
        - 0.5
        * tf.reduce_sum(-t_log_var + tf.exp(t_log_var) + tf.square(t_mean) - 1, axis=1)
    )
    return -vlb


def create_encoder(input_dim):
    # Encoder network.
    # We instantiate these layers separately so as to reuse them later
    encoder = Sequential(name="encoder")
    encoder.add(InputLayer([input_dim]))
    encoder.add(Dense(intermediate_dim, activation="relu"))
    encoder.add(Dense(2 * latent_dim))
    return encoder


def create_decoder(input_dim):
    # Decoder network
    # We instantiate these layers separately so as to reuse them later
    decoder = Sequential(name="decoder")
    decoder.add(InputLayer([input_dim]))
    decoder.add(Dense(intermediate_dim, activation="relu"))
    decoder.add(Dense(original_dim, activation="sigmoid"))
    return decoder


# Sampling from the distribution
#     q(t | x) = N(t_mean, exp(t_log_var))
# with reparametrization trick.
def sampling(args):
    """Returns sample from a distribution N(args[0], diag(args[1]))
    
    The sample should be computed with reparametrization trick.
    
    The inputs are tf.Tensor
        args[0]: (batch_size x latent_dim) mean of the desired distribution
        args[1]: (batch_size x latent_dim) logarithm of the variance vector of the desired distribution
    
    Returns:
        A tf.Tensor of size (batch_size x latent_dim), the samples.
    """
    t_mean, t_log_var = args
    # YOUR CODE HERE
    epsilon = K.random_normal(t_mean.shape)
    z = epsilon * K.exp(0.5 * t_log_var) + t_mean
    return z


batch_size = 100
original_dim = 784  # Number of pixels in MNIST images.
latent_dim = 3  # d, dimensionality of the latent code t.
intermediate_dim = 128  # Size of the hidden layer.
epochs = 20

x = Input(batch_shape=(batch_size, original_dim))

encoder = create_encoder(original_dim)

get_t_mean = Lambda(lambda h: h[:, :latent_dim])
get_t_log_var = Lambda(lambda h: h[:, latent_dim:])
h = encoder(x)
t_mean = get_t_mean(h)
t_log_var = get_t_log_var(h)

t = Lambda(sampling)([t_mean, t_log_var])

decoder = create_decoder(latent_dim)
x_decoded_mean = decoder(t)
