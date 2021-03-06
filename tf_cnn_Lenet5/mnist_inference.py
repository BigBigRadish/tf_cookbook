# -*- coding: utf-8 -*-
'''
Created on 2018年9月4日

@author: Zhukun Luo
Jiangxi university of finance and economics
'''
import tensorflow as tf
#配置神经网络参数
INPUT_NODE = 784
OUTPUT_NODE = 10
IMAGE_SIZE=28
NUM_CHANNELS=1
NUM_LABELS=10
#第一层卷积层的尺寸和深度
CONV1_DEEP=32
CONV1_SIZE=5
#第二层卷积层的尺寸和深度
CONV2_DEEP=64
CONV2_SIZE=5
#全连接节点个数
FC_SIZE=512

def inference(input_tensor,train, regularizer):
    with tf.variable_scope('layer1-conv1'):
        conv1_weights = tf.get_variable("weights",[CONV1_SIZE, CONV1_SIZE,NUM_CHANNELS,CONV1_DEEP],initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv1_biases = tf.get_variable("biases", [CONV1_DEEP], initializer=tf.constant_initializer(0.0))
    #使用边长为5，深度为32的过滤器，且步长为1，使用全0填充
        conv1=tf.nn.conv2d(input_tensor,conv1_weights,strides=[1,1,1,1],padding='SAME')
        relu1=tf.nn.relu(tf.nn.bias_add(conv1,conv1_biases))

    with tf.name_scope('layer2-pool1'):
        pool1=tf.nn.max_pool(relu1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
    #声明第三层卷积层变量并实现前向传播过程，这一层的输入为14*14*32
    #输出为14*14*64
    with tf.variable_scope('layer3-conv1'):
        conv2_weights = tf.get_variable("weights",[CONV2_SIZE, CONV2_SIZE,NUM_CHANNELS,CONV2_DEEP],initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv2_biases = tf.get_variable("biases", [CONV2_DEEP], initializer=tf.constant_initializer(0.0))
    #使用边长为5，深度为32的过滤器，且步长为1，使用全0填充
        conv2=tf.nn.conv2d(input_tensor,conv2_weights,strides=[1,1,1,1],padding='SAME')
        relu2=tf.nn.relu(tf.nn.bias_add(conv2,conv2_biases))
    with tf.name_scope('layer4-pool2'):
        pool2=tf.nn.max_pool(relu2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')#第四层输出7*7*64
    pool_shape=pool2.get_shape().as_list()
    #计算将矩阵拉直为向量的长度#
    #pool_shape[0]为batch中数据的个数
    nodes=pool_shape[1]*pool_shape[2]*pool_shape[3]
    #通过tf.reshape()将第四层拉升为向量
    reshaped=tf.reshape(pool2,[pool_shape[0],nodes])
    #声明第五层全连接层的变量并实现前向传播过程。这一层的输入是拉直后的向量。向量长度为3136，输出是一组长度为512的向量。引入dropout的概念。dropout在训练时会随机将部分节点的输出改为0.dropout一般在全连接层而不是卷积或者池化层
    with tf.variable_scope('layer5-fc1'):
        fc1_weights=tf.get_variable('weights',[nodes,FC_SIZE],initializer=tf.truncated_normal_initializer(stddev=0.1))
        #只有全连接层的权重需要加入正则化
        if regularizer != None:
            tf.add_to_collection('losses', regularizer(fc1_weights))
        fc1_bias=tf.get_variable('biases', [FC_SIZE],initializer=tf.constant_initializer(0.1))
        fc1=tf.nn.relu(tf.matmul(reshaped,fc1_weights)+fc1_bias)
        if train:
            fc1=tf.nn.dropout(fc1,0.5)
    #第六层，输入长度为512，输出为10
    with tf.variable_scope('layer6-fc2'):
        fc2_weights=tf.get_variable('weights',[FC_SIZE,NUM_LABELS],initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None:
            tf.add_to_collection('losses', regularizer(fc2_weights))
        fc2_bias=tf.get_variable('biases', [NUM_LABELS],initializer=tf.constant_initializer(0.1))
        logit=tf.matmul(fc1,fc2_weights)+fc2_bias        
    return logit   