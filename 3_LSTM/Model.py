# -*- coding: utf-8 -*-

import tensorflow as tf
import embLayer
import WordLSTMLayer
import Pooling
import Fulllayer
import softmaxLayer
from config import *

# 定义神经网络的前向传播过程。
def buildModel(docs,voc,label,istrain):

    embeding_layer = embLayer.embLayer(docs,voc,label, HIDDEN_SIZE)

    LSTM_layer = WordLSTMLayer.LSTMLayer(embeding_layer,HIDDEN_SIZE,"word",istrain)

    pooling = Pooling.meanpooling(LSTM_layer)

    fullLayer = Fulllayer.fullLayer(pooling, HIDDEN_SIZE)

    softlayer=softmaxLayer.softLayer(fullLayer, HIDDEN_SIZE,OUTPUT_CLASS)

    return softlayer

