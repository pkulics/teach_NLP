# -*- coding: utf-8 -*-

import tensorflow as tf
from config import *
# 定义神经网络相关的参数



def get_weight_variable(shape,name):
    weights = tf.get_variable(name, shape, initializer=tf.random_uniform_initializer)

    return weights

def softLayer(input,hidden_size,out_class):
    '''softmax层'''
    weight = get_weight_variable([hidden_size, out_class],"softmax")
    bias = tf.get_variable("bias_s", [out_class])
    logits = tf.matmul(input, weight)+bias

    return logits
