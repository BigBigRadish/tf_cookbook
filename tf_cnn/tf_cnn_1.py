# -*- coding: utf-8 -*-
'''
Created on 2018年9月5日

@author: Zhukun Luo
Jiangxi university of finance and economics
'''
import tensorflow as tf
#经过tf.get_variable的方式创建过滤器的权重变量和偏置项变量。
#cnn的参数只和过滤器的尺寸、深度以及当前层节点矩阵的深度有关，所以
#这里声明的参数变量是一个思维矩阵，前面两个维度代表了过滤器的尺寸，第三个表示当前层的深度，第四个维度表示过滤器的深度
filter_weight=tf.get_variable('weight',[5,5,3,16],initializer=tf.truncated_normal_initializer(stddev=0.1))
#和卷积层的权重类似，当前层矩阵上不同位置的偏置项也是共享的，所以总共有下一层深度个不同的偏置项。16也是下一层的深度。
bias=tf.get_variable('bias',[16],initializer=tf.truncated_normal_initializer(0.1))
#tf.nn.conv2d提供了一个非常方便的函数来实现卷积层前向传播的算法，
#这个函数的第一个输入为当前层的节点矩阵。注意这个矩阵是一个四维矩阵，后面三个维度对应一个节点矩阵，第一维对应一个输入的batch。比如在输入层，input[0,:,:,:]表示第一张图片，input[0,:,:,:]表示第二张图片
#第二个参数提供了卷积层的权重，第三个参数为不同维度的步长。第三个参数第一位和最后一位的数字必须是1.因为卷积层的步长只对矩阵的长和宽有效。最后一个参数是填充的方法。有SAME和VALID两种选择。其中前者表示全为0填充，后者表示不填充。
conv=tf.nn.conv2d(input,filter_weight,strides=[1,1,1,1],padding='SAME')
#tf.nn.bias_add提供了一个方便的函数给每一个节点加上偏置项。
bias=tf.nn.bias_add(conv,bias)
actived_conv=tf.nn.relu(bias)#去线性化