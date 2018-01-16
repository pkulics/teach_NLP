# -*- coding: utf-8 -*-

import tensorflow as tf
from config import *
import numpy
# 定义神经网络相关的参数



def get_weight_variable(shape,name):
    weights = tf.get_variable(name, shape,
                              initializer=tf.random_uniform_initializer)
    return weights

def fullLayer(input,hidden_size):
    '''全连接层'''
    weight = get_weight_variable([hidden_size, hidden_size],"weight")
    bias = tf.get_variable("bias", [hidden_size])
    logits = tf.nn.tanh(tf.matmul(input, weight)+bias)

    return logits

