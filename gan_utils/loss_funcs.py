import tensorflow as tf

def discriminator_loss(true_labels, gen_labels):
    loss = tf.reduce_sum(tf.log(true_labels + 1e-19) + tf.log(1 - gen_labels + 1e-19), axis=-1)
    return -tf.reduce_mean(loss)

def generator_loss(gen_labels):
    loss = tf.reduce_sum(tf.log(gen_labels + 1e-19), axis=-1)
    return -tf.reduce_mean(loss)
