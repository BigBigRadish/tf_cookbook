# -*- coding: utf-8 -*-
'''
Created on 2018年9月3日

@author: Zhukun Luo
Jiangxi university of finance and economics
'''
#训练我的第一个神经网络
import tensorflow as tf
import numpy as np
from numpy.random import RandomState#通过numpy工具包生成模拟数据集
#定义训练数据batch大小
batch_size=8
#定义神经网络的参数，这里还是用前向网络传播结构
w1=tf.Variable(tf.random_normal([2,1], stddev=1,seed=1))
x=tf.placeholder(tf.float32,shape=(None,2),name="x-input")#3组数据,使用None可以方便训练不同batch的数据，大量数据容易造成堆栈溢出
y_=tf.placeholder(tf.float32,shape=(None,1),name="y-input")
y=tf.matmul(x,w1)
#定义预测多了和少了的成本
loss_less=10
loss_more=1
loss=tf.reduce_sum(tf.where(tf.greater(y,y_), (y-y_)*loss_more, (y_-y)*loss_less))
train_step=tf.train.AdamOptimizer(0.001).minimize(loss)
#通过随机数生成一个模拟数据集
rdm=RandomState(1)
dataset_size=128
X=rdm.rand(dataset_size,2)
Y=[[x1+x2+rdm.rand()/10.0-0.05] for (x1,x2) in X]
with tf.Session() as sess:
    init_op=tf.global_variables_initializer()#初始化变量
    sess.run(init_op)
    print(sess.run(w1))
    '''
    训练之前权重值
    w1=[[-0.81131822]
 [ 1.48459876]]
    '''
    #设定训练的轮数
    steps=10000
    for i in range(steps):
        #每次选取batch_size个样本进行训练
        start=(i *batch_size)%dataset_size
        end=min(start+batch_size,dataset_size-1)
        #通过选取的样本训练神经网络并更新参数
        sess.run(train_step,feed_dict={x:X[start:end],y_:Y[start:end]})
        if i%1000==0:
            #每隔一段时间输出交叉熵
            total_cross_entropy=sess.run(loss,feed_dict={x:X,y_:Y})
            print("After %d training steps,cross entropy on all dataset is %g" % (i,total_cross_entropy))#交叉熵越小说明与真实结果差距越小
    print(sess.run(w1))
'''
    训练之后：
    After 0 training steps,cross entropy on all dataset is 763.879
After 1000 training steps,cross entropy on all dataset is 171.785
After 2000 training steps,cross entropy on all dataset is 87.1666
After 3000 training steps,cross entropy on all dataset is 56.67
After 4000 training steps,cross entropy on all dataset is 20.9553
After 5000 training steps,cross entropy on all dataset is 6.46666
After 6000 training steps,cross entropy on all dataset is 6.46687
After 7000 training steps,cross entropy on all dataset is 6.46703
After 8000 training steps,cross entropy on all dataset is 6.46622
After 9000 training steps,cross entropy on all dataset is 6.46611
[[ 1.01997852]
 [ 1.04333842]]
'''