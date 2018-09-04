# -*- coding: utf-8 -*-
'''
Created on 2018年9月4日

@author: Zhukun Luo
Jiangxi university of finance and economics
'''
#滑动平均模型
#tf.train.ExponentialMovingAverage实现滑动平均
#初始化模型需要提供一个衰减率（decay），用于控制模型更新的速度
#对每一个变量都维护一个影子变量，这个影子变量的初始值就是相应变量的初始值，每次运行变量时都会更新：shadow——variable=decay*shadow_variable+(1-decay)*variable
import tensorflow as tf
#定义一个变量用于计算滑动平均，这个变量的初始值是0，手动指定变量的类型是tf.float32,因为变量必须是实数型
v1=tf.Variable(0,dtype=tf.float32)
#这里的step变量模拟神经网络中的轮数，阔以用来动态控制衰减率
step=tf.Variable(0,trainable=False)
#定义一个滑动平均的类，初始化给定衰减率（0.99）和控制衰减率的变量step。
ema=tf.train.ExponentialMovingAverage(0.99,step)
#定义一个更新变量滑动平均的操作
maintrain_average_op=ema.apply([v1])
with tf.Session() as sess:
    init_op=tf.global_variables_initializer()
    sess.run(init_op)
    #通过ema.average(v1)获取滑动平均后的值，初始化都为0
    print(sess.run([v1,ema.average(v1)]))
    #更新v1的值为5
    print(sess.run(tf.assign(v1,5)))#给变量赋值
    sess.run(maintrain_average_op)
    print(sess.run([v1,ema.average(v1)]))
    #更新step的值为1000
    sess.run(tf.assign(step,10000))#用来设置动态衰减率，算出decay，初始衰减很快，后面后满
    print(sess.run(tf.assign(v1,10)))#给变量赋值
    sess.run(maintrain_average_op)
    print(sess.run([v1,ema.average(v1)]))    