# -*- coding: utf-8 -*-
'''
Created on 2018年9月3日

@author: Zhukun Luo
Jiangxi university of finance and economics
'''
import tensorflow as tf
w1=tf.Variable(tf.random_normal([2,3], stddev=1))
w2=tf.Variable(tf.random_normal([3,1], stddev=1))
#placeholder提供数据输入的地方，不用定义大量常量，且不用定义输入维度，placeholder阔以自动计算维度
# x=tf.placeholder(tf.float32,shape=(1,2),name="input")#一行两列的输入样例
# a=tf.matmul(x,w1)
# y=tf.matmul(a,w2)
# sess=tf.Session()
# init_op=tf.global_variables_initializer()
# sess.run(init_op)
# print(sess.run(y,feed_dict={x:[[0.7,0.9]]}))#输出：[[ 1.32559538]]
x=tf.placeholder(tf.float32,shape=(3,2),name="input")#3组数据
a=tf.matmul(x,w1)
y=tf.matmul(a,w2)
sess=tf.Session()
init_op=tf.global_variables_initializer()
sess.run(init_op)
print(sess.run(y,feed_dict={x:[[0.7,0.9],[0.1,0.4],[0.5,0.9]]}))
#使用sigmoid函数将y转换为0-1之间的数值，转换后y代表正样本的概率，1-y代表负样本的概率
y=tf.sigmoid(y)
#定义损失函数来刻画预测值与真实值的差距
cross_entropy=-tf.reduce_mean(y* tf.log(tf.clip_by_value(y,1e-10,1.0))+(1-y)*tf.log(tf.clip_by_value(1-y,1e-10,1.0)))#定义损失函数
learning_rate=0.001#定义学习率
train_rate=tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)#定义反向传播优化函数
#以上为定义一个神经网络的基本步骤