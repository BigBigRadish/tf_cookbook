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
w1=tf.Variable(tf.random_normal([2,3], stddev=1,seed=1))
w2=tf.Variable(tf.random_normal([3,1], stddev=1,seed=1))
#定义数据的输入
x=tf.placeholder(tf.float32,shape=(None,2),name="x-input")#3组数据,使用None可以方便训练不同batch的数据，大量数据容易造成堆栈溢出
y_=tf.placeholder(tf.float32,shape=(None,1),name="y-input")
#定义神经网络前向传播过程
a=tf.matmul(x,w1)
y=tf.matmul(a,w2)
#定义损失函数和反向传播算法
y=tf.sigmoid(y)
#定义损失函数来刻画预测值与真实值的差距
cross_entropy=-tf.reduce_mean(y_* tf.log(tf.clip_by_value(y,1e-10,1.0))+(1-y)*tf.log(tf.clip_by_value(1-y,1e-10,1.0)))#定义损失函数
learning_rate=0.001#定义学习率
train_rate=tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)#定义反向传播优化函数

#通过随机数生成一个模拟数据集
rdm=RandomState(1)
dataset_size=128
X=rdm.rand(dataset_size,2)
Y = [[int(x1+x2<1)] for (x1,x2) in X]#创建标签
print(Y)
#创建一个会话来运行tensorflow程序
with tf.Session() as sess:
    init_op=tf.global_variables_initializer()#初始化变量
    sess.run(init_op)
    print(sess.run(w1))
    print(sess.run(w2))
    '''
    训练之前权重值
    w1=[[-0.81131822  1.48459876  0.06532937]
     [-2.4427042   0.0992484   0.59122431]]
    w2=[[-0.81131822]
     [ 1.48459876]
     [ 0.06532937]]
    '''
    #设定训练的轮数
    steps=5000
    for i in range(steps):
        #每次选取batch_size个样本进行训练
        start=(i *batch_size)%dataset_size
        end=min(start+batch_size,dataset_size-1)
        #通过选取的样本训练神经网络并更新参数
        sess.run(train_rate,feed_dict={x:X[start:end],y_:Y[start:end]})
        if i%1000==0:
            #每隔一段时间输出交叉熵
            total_cross_entropy=sess.run(cross_entropy,feed_dict={x:X,y_:Y})
            print("After %d training steps,cross entropy on all dataset is %g" % (i,total_cross_entropy))#交叉熵越小说明与真实结果差距越小
    print(sess.run(w1))
    print(sess.run(w2))      
    

