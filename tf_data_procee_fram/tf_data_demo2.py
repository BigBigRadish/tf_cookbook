# -*- coding: utf-8 -*-
'''
Created on 2018年9月9日

@author: Zhukun Luo
Jiangxi university of finance and economics
'''
#在自然语言处理任务中，训练数据通常是以每行一条数据的形式存在文本文件中，这是阔以用TextLineDataset来更方便读取数据
import tensorflow as tf
#从文本文件中创建数据集。假定每行文字是一个训练例子
input_files=['file1','file2']
dataset=tf.data.TextLineDataset(input_files)
#定义迭代器用于遍历数据
iterator=dataset.make_one_shot_iterator()
x=iterator.get_next()
with tf.Session() as sess:
    for i in range(3):
        print(sess.run(x))
#上卖弄例子都是用的make_one_shot_iterator来遍历数据集，在使用这个函数时，必须所有参数都得确定，所以就不用初始化。
#如果用placeholder来初始化数据集，需要用到initializable_iterator