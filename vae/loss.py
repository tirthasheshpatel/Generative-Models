from __future__ import print_function, division, absolute_import

import tensorflow as tf

class BinomialVLB(object):
    def __init__(self, name="BinomialVariationalLowerBound"):
        self.name = name

    @staticmethod
    def unsigned_call(x, x_recon, loc, log_var):
        KL_loss = - 0.5 * tf.reduce_sum( -log_var + tf.math.exp( log_var ) + tf.square( loc ) - 1, axis=-1 )
        cross_entropy_loss = tf.reduce_sum( x * tf.math.log( x_recon+1e-10 ) + (1-x) * tf.math.log( 1-x_recon+1e-10 ), axis=[-3, -2, -1] )
        vlb = tf.reduce_mean(cross_entropy_loss + KL_loss)
        return vlb

class NormalVLB(object):
    def __init__(self, name="NormalVariationalLowerBound"):
        self.name = name

    @staticmethod
    def unsigned_call(x, x_recon, loc, log_var):
        KL_loss = - 0.5 * tf.reduce_sum( -log_var + tf.math.exp( log_var ) + tf.square( loc ) - 1, axis=-1 )
        reconstruction_loss = tf.reduce_sum( tf.square( x - x_recon ), axis=[-3, -2, -1] )
        vlb = tf.reduce_mean(reconstruction_loss + KL_loss)
        return vlb
