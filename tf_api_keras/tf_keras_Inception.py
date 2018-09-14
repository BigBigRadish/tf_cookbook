# -*- coding: utf-8 -*-
'''
Created on 2018年9月14日

@author: Zhukun Luo
Jiangxi university of finance and economics
'''
#keras api 和tensorflow混用
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist_data=input_data.read_data_sets('E:/workplace/tf_cookbook/tf_2_mnist_best/mnist',one_hot=True)
x=tf.placeholder(tf.float32, shape=(None,784))
y_=tf.placeholder(tf.float32,shape=(None,10))
#直接使用tensorflow里面的keras定义网络结构
net=tf.keras.layers.Dense(500,activation='relu')(x)
y=tf.keras.layers.Dense(10,activation='softmax')(net)
#定义损失函数和优化方法，注意这里可以混用keras中的api和原生态tensorflow中的api
loss=tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y_,y))
train_step=tf.train.GradientDescentOptimizer(0.5).minimize(loss)
#定义预测的正确率作为指标
acc_value=tf.reduce_mean(tf.keras.metrics.categorical_accuracy(y_,y))
#使用原生态的tensorflow训练模型，可以有效使用分布式
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    for i in range(10000):
        xs,ys=mnist_data.train.next_batch(100)
        _,loss_value=sess.run([train_step,loss],feed_dict={x:xs,y_:ys})
        if i%1000==0:
            print('after %d training steps,loss on training batch is %g'%(i,loss_value))
    print(acc_value.eval(feed_dict={x:mnist_data.test.images,y_:mnist_data.test.labels}))