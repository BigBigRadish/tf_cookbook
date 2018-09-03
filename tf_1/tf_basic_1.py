# -*- coding: utf-8 -*-
'''
Created on 2018年9月1日

@author: Zhukun Luo
Jiangxi university of finance and economics
'''
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow  as tf 
a= tf.constant([1.0,2.0],name='a')
b= tf.constant([1.0,3.0],name='b')
result=a+b#一个引用
sess=tf.Session()
print(sess.run(result))
sess.close()
#创建一个会话，使用python的上下文管理器来管理会话
with tf.Session() as sess:#创建一个上下文,解决资源泄露
    sess.run(result)
sess=tf.Session()
with sess.as_default():
    print(result.eval())#通过默认绘话输出张量的取值
sess=tf.InteractiveSession()#直接构建默认绘话的函数
print(result.eval())
sess.close()

    