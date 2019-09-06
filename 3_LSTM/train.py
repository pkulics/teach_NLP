# -*- coding: utf-8 -*-

import tensorflow as tf
from Dataset import *
import Model
import numpy
from config import *
import numpy as np


def train(trainset,devset,trainset_label,devset_label,voc):
    # 定义输入输出placeholder。
    tra_docs = tf.placeholder(tf.int64, [None,None], name='x-input')
    lab = tf.placeholder(tf.int64, [None], name='y-input')
    '''
    调用Model里的模型实现初始化
    '''
    y=Model.buildModel(tra_docs,voc,lab,istrain=1)
    '''
    定义过程中的各种参数
    '''
    global_step = tf.Variable(0, trainable=False)
    #定义损失函数、学习率
    correct_prediction = tf.equal(tf.argmax(tf.nn.softmax(y,1), 1) , lab)
    correct = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=lab)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean

    err = tf.argmax(y, 1) - lab
    mse = tf.reduce_sum(err * err)
    train_step = tf.train.AdagradOptimizer(LEARNING_RATE_BASE).minimize(loss, global_step=global_step)


    # 初始化Tensorflow持久化类
    saver = tf.train.Saver()
    with tf.Session() as sess:

        tf.global_variables_initializer().run()
        acc_final = 0.0
        epoch = len(trainset.docs) / BATCH_SIZE - 1
        for i in range(TRAINING_STEPS):
            n=0
            train_batch_list = np.random.randint(epoch,size=TRAINING_SIZE)
            print "**************************************************************************************"
            for j in train_batch_list:
                n+=1
                docs = trainset.docs[j * BATCH_SIZE:(j + 1) * BATCH_SIZE]
                label = trainset_label.label[j * BATCH_SIZE:(j + 1) * BATCH_SIZE]
                steploss = sess.run(train_step,feed_dict={tra_docs:docs,lab:label})
                print 'step loss',n,':',steploss

            acc = 0.0
            tot = 0.0
            mse_final = 0.0
            cor = 0.0
            for k in range(epoch-1):
                docs_test = devset.docs[k * BATCH_SIZE:(k + 1) * BATCH_SIZE]
                label_test = devset_label.label[k * BATCH_SIZE:(k + 1) * BATCH_SIZE]

                accuracy_score,mse_score,cor_score= sess.run([accuracy,mse,correct], feed_dict={tra_docs: docs_test, lab: label_test})
                acc += accuracy_score
                mse_final +=mse_score
                cor +=cor_score
                tot+=len(label_test)

            print 'step accuracy1: ',i, ':',acc / float(epoch - 1),'accuracy2:',float(cor)/float(tot),'RMSE:',numpy.sqrt(float(mse_final)/float(tot))
            saver.save(sess, MODEL_SAVE_PATH + MODEL_NAME)
            if float(cor)/float(tot)>=acc_final:
                acc_final = float(cor)/float(tot)
                saver.save(sess, MODEL_SAVE_PATH+MODEL_NAME)
                print 'best model saved!'

        print '************train stop,best model saved!good luck!*************'
        print 'final accuracy',acc_final


def main(argv=None):

    voc = Wordlist('./data/wordlist.txt')
    trainset = Dataset('./data/train.txt', voc, BATCH_SIZE)
    devset = Dataset('./data/train.txt', voc, BATCH_SIZE)
    trainset_label = Label('./data/train_label.txt')
    devset_label = Label('./data/train_label.txt')
    print "data loaded!"

    train(trainset,devset,trainset_label,devset_label,voc)

if __name__ == '__main__':
    tf.app.run()
