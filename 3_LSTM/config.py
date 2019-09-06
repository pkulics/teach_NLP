# -*- coding: utf-8 -*-

# 配置神经网络的参数
'''batch大小'''
BATCH_SIZE = 32
'''初始学习率'''
LEARNING_RATE_BASE = 0.001
'''学习率衰减系数'''
LEARNING_RATE_DECAY = 0.95
'''L2正则化系数'''
REGULARAZTION_RATE = 0.0001
'''训练轮数'''
TRAINING_STEPS = 3
'''训练数据batch数'''
TRAINING_SIZE = 2
'''滑动平均参数'''
MOVING_AVERAGE_DECAY = 0.95
'''隐藏层节点数'''
HIDDEN_SIZE = 300
'''分类类别数'''
OUTPUT_CLASS = 10
'''LSTM 不被dropout 比率（保留参数比）'''
LSTM_KEEP_PROB = 0.9
'''LSTM层数'''
LSTM_NUM_LAYERS = 2
'''模型保存的路径和文件名'''
MODEL_SAVE_PATH = "model/"
MODEL_NAME = "model.ckpt"
'''数据集名称'''
dataname = 'SST'