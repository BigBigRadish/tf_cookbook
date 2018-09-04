# -*- coding: utf-8 -*-
'''
Created on 2018年9月4日

@author: Zhukun Luo
Jiangxi university of finance and economics
'''
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.ops.init_ops import Initializer
mnist=input_data.read_data_sets('./mnist/',one_hot=True)
print("Training data size:",mnist.train.num_examples)
print("validating data size:",mnist.validation.num_examples)
print("testing data size:",mnist.test.num_examples)
print("Example training data:",len(mnist.train.images))#是个list
batch_size=100
xs,ys=mnist.train.next_batch(batch_size)
#从train的集合中选区batch_size个训练数据
print("X shape:",xs.shape)    
print("Y shape:",ys.shape) 
#TENSORFLOW变量管理
#下面这两个变量定义是等价的
v= tf.get_variable("v",shape=[1],initializer=tf.constant_initializer(1.0))#变量名称必须填写，避免无意识的变量复用
v=tf.Variable(tf.constant(1.0,shape=[1],name="v"))
#通过tf.Variable_scope函数控制tf.get_variable()函数已经创建的变量
with tf.variable_scope("foo"):#，命名空间为foo
    v=tf.get_variable("v",[1],initializer=tf.constant_initializer(1.0))
#命名空间已经存在名字为v的变量，所以下面代码将会报错
# with tf.variable_scope("foo"):#，命名空间为foo
#     v=tf.get_variable("v",[1],initializer=tf.constant_initializer(1.0))
#在生成上下文管理器时，将参数reuse设置为True，这样tf.get_variable函数直接获取已经神明的变量
with tf.variable_scope("foo",reuse=True):
    v1=tf.get_variable("v",[1])
    print(v==v1)
#可以使用这种方式进行程序嵌套