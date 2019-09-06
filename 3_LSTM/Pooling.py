# -*- coding: utf-8 -*-
import tensorflow as tf


def meanpooling(input):
    with tf.variable_scope("pooling_layer"):
        meanpooling = tf.reduce_mean(input,axis=1)
    return meanpooling

