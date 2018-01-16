# -*- coding: utf-8 -*-
import tensorflow as tf


def embLayer(input,vocab_size,label,hidden_size):
    docsbatch = input
    with tf.variable_scope('embdding'):
        emb = tf.get_variable('emb', [vocab_size.size, 300], trainable=True)#
        embedding_input = tf.nn.embedding_lookup(emb, docsbatch)
    return embedding_input

