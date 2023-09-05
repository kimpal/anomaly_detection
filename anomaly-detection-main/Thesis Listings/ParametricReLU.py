import tensorflow as tf
import torch.nn as nn

# Parametric Rectified Linear Unit - LeakyReLU
def parametric_relu(_x):
    alphas = tf.get_variable('alpha', _x.get_shape()[-1],
                            initializer=tf.constant_initializer(0.0),
                            dtype=tf.float32)
    pos = tf.nn.relu(_x)
    neg = alphas * (_x - abs(_x)) * 0.5

    return pos + neg