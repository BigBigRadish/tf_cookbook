# -*- coding: utf-8 -*-
'''
Created on 2018年9月4日

@author: Zhukun Luo
Jiangxi university of finance and economics
'''
import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import mnist_inference
import mnist_train
import numpy as np
#每十秒加载一次最新的模型，并在测试数据上测试最新模型的正确率

EVAL_INTERVAL_SECS = 10
def evaluate(mnist):
    with tf.Graph().as_default() as g:
        test_size=mnist.validation.images.size
        x = tf.placeholder(tf.float32, [None, mnist_inference.IMAGE_SIZE,mnist_inference.IMAGE_SIZE,mnist_inference.NUM_CHANNELS], name='x-input')
        y_ = tf.placeholder(tf.float32, [None, mnist_inference.OUTPUT_NODE], name='y-input')
        print(np.array(mnist.validation.images))
        xs=np.reshape(np.array(mnist.validation.images),(mnist.validation.images.size, mnist_inference.IMAGE_SIZE,mnist_inference.IMAGE_SIZE,mnist_inference.NUM_CHANNELS))
        validate_feed = {x : xs, y_:mnist.validation.labels}
        #直接通过调用封装好的函数来计算前向传播的结果。因为测试时不关注正则化损失的值，所以这里用于计算正则化损失的函数被设置为None
        y = mnist_inference.inference(x, False,None)
        correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        variable_averages = tf.train.ExponentialMovingAverage(mnist_train.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)
        while True:
            with tf.Session() as sess:
                ckpt = tf.train.get_checkpoint_state(mnist_train.MODEL_SAVE_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    accuracy_score = sess.run(accuracy, feed_dict = validate_feed)
                    print("After %s training step(s), validation accuracy = %g" % (global_step, accuracy_score))
                else:
                    print('No checkpoint file found')
                    return
                #time.sleep(EVAL_INTERVAL_SECS)
def main(argv=None):
    mnist = input_data.read_data_sets('./mnist/', one_hot=True)
    evaluate(mnist)
if __name__ == '__main__':
    tf.app.run()