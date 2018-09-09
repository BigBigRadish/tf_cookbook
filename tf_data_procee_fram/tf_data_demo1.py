# -*- coding: utf-8 -*-
'''
Created on 2018年9月9日

@author: Zhukun Luo
Jiangxi university of finance and economics
'''
#与队列一样，数据集也是图上的一个节点
import tensorflow as tf 
#从一个数组创建数据集
input_data=[1,2,3,5,8]
dataset=tf.data.Dataset.from_tensor_slices(input_data)
#定义一个迭代器用于遍历数据集，因为上面的数据没有用placeholder作为输如参数，所以这里可以使用简单的one_hot_iterator
iterator=dataset.make_one_shot_iterator()
x=iterator.get_next()
y=x*x
with tf.Session() as sess:
    for i in range(len(input_data)):
        print(sess.run(y))