# -*- coding: utf-8 -*-
'''
Created on 2018年9月3日

@author: Zhukun Luo
Jiangxi university of finance and economics
'''
import tensorflow as tf
'''
#目前tensorflow提供了7种不同的激活函数，relu，sigmoid，tanh，比较常用。tensorflow也支持自定义激活函数
a= tf.nn.relu(tf.matmul(x,w1)+biases1)
y=tf.nn.relu(tf.matmul(a,w2)+biases2)
'''
#经典损失函数;
#交叉熵刻画了两个概率分布之间的距离，原本用来估算平均编码长度
#cross_entropy=-tf.reduce_mean(y_* tf.log(tf.clip_by_value(y,1e-10,1.0))+(1-y)*tf.log(tf.clip_by_value(1-y,1e-10,1.0)))#定义损失函数
v= tf.constant([[1.0,2.0,3.0],[4.0,5.0,6.0]])
sess=tf.Session()
with sess.as_default():
    print(tf.clip_by_value(v,2.5,4.5).eval())#clip_by_value()限制数值范围
    print(tf.log(v).eval())#求对数
#交叉熵一般与softmax回归一起使用
#cross_entropy=tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=y)
#回归问题常用的是均方误差
#mse=tf.reduce_mean(tf.square(y_-y))
v1= tf.constant([1.0,2.0,3.0,4.0])
v2= tf.constant([4.0,3.0,2.0,1.0])
with tf.Session().as_default():
    print(tf.greater(v1, v2).eval())#》
    print(tf.where(tf.greater(v1, v2),v1,v2).eval())#哪一个大返回哪一个
    
    