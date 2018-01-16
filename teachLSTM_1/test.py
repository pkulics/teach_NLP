# -*- coding: utf-8 -*-

import tensorflow as tf
from Dataset import *
import Model
from config import *




def evaluate(testset,testset_label,voc):
    with tf.Graph().as_default() as g:
        # 定义输入输出的格式

        tra_docs = tf.placeholder(tf.int32, [None, None], name='x-input')
        lab = tf.placeholder(tf.int64, [None], name='y-input')
        y = Model.buildModel(tra_docs, voc, lab, None,istrain=0)

        correct_prediction = tf.equal(tf.argmax(y,1), lab)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY)
        variable_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variable_to_restore)

        with tf.Session() as sess:
            saver.restore(sess,  MODEL_SAVE_PATH+MODEL_NAME)
            acc = 0
            accuracy_score = sess.run(accuracy, feed_dict={tra_docs:testset.docs , lab: testset_label.label})
            acc +=accuracy_score
            print 'accuracy',accuracy_score,'acc',acc


def main(argv=None):
    voc = Wordlist('./data/wordlist.txt')
    testset = Dataset('./data/train.txt', voc, BATCH_SIZE)
    testset_label = Label('./data/train_label.txt')
    print "data loaded!"

    evaluate( testset, testset_label, voc)

if __name__ == '__main__':
    tf.app.run()