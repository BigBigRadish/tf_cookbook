# -*- coding: utf-8 -*-
'''
Created on 2018年9月26日

@author: Zhukun Luo
Jiangxi university of finance and economics
'''
import tensorflow as tf 
#定义一个简单的计算图，实现向量加法操作
input1=tf.constant([1.0,2.0,3.0],name='input1')
input2=tf.Variable(tf.random_uniform([3]),name='input2')
output=tf.add_n([input1,input2],name='add')
#生成一个写日志的writer，并将当前的temnsorflow计算图写入日志
writer=tf.summary.FileWriter('log',tf.get_default_graph())#需要pip install tb-nightly
#解决方案：https://blog.csdn.net/handsome_for_kill/article/details/80269595
