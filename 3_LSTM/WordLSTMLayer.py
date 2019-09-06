# -*- coding: utf-8 -*-

import tensorflow as tf
from config import *


def LSTMLayer(input,hidden_size,name,istrain):
    '''定义LSTM'''
    with tf.variable_scope(name+'lstm',initializer=tf.random_uniform_initializer):
        lstmcell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size)
        if istrain:
            lstmcell = tf.nn.rnn_cell.DropoutWrapper(lstmcell,output_keep_prob=LSTM_KEEP_PROB)
            initial_state = lstmcell.zero_state(tf.shape(input)[0], tf.float32)
    with tf.variable_scope(name+"RNN",initializer=tf.random_uniform_initializer):
        cell_output, state = tf.nn.dynamic_rnn(cell=lstmcell, time_major=False,inputs=input, dtype=tf.float32,initial_state=initial_state)
        lstmoutput = cell_output

    return lstmoutput


