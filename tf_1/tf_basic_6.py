# -*- coding: utf-8 -*-
'''
Created on 2018年9月3日

@author: Zhukun Luo
Jiangxi university of finance and economics
'''
#学习率的迭代更新
import tensorflow as tf
global_step=tf.Variable(0)
#通过exponential_decay函数生成学习率、
learning_rate=tf.train.exponential_decay(0.1,global_step,100,0.96,staircase=True)
#使用指数衰减的学习率，在mininmize函数中传入global_step将自动更新
#global_step参数，从而使学习率得到相应更新
#learning_step=tf.train.GradientDescentOptimizer(learning_rate).minimize(my loss, global_step=global_step)
#带L2损失函数的定义
# w=tf.Variable(tf.random_normal([2,1],stddev=1,seed=1))
# y=tf.matmul(x,w)
# loss=tf.reduce_mean(tf.square(y_-y))+tf.contrib.layers.l2_regularizer(lambda)(w)#偏置项
weights=tf.constant([[1.0,-2.0],[-3.0,4.0]])
with tf.Session() as sess:
    #输出为（1--2+-3+4）*0.5=5
    print(sess.run(tf.contrib.layers.l1_regularizer(.5)(weights)))#l1
    print(sess.run(tf.contrib.layers.l2_regularizer(.5)(weights)))#l2