import tensorflow as tf

def discriminator_loss(true_labels, gen_labels):
    loss = tf.reduce_mean(tf.log(true_labels + 1e-8) + tf.log(1 - gen_labels + 1e-8))
    return -loss

def generator_loss(gen_labels):
    loss = tf.reduce_mean(tf.log(gen_labels + 1e-8))
    return -loss
