# -*- coding: utf-8 -*-
'''
Created on 2018年9月4日

@author: Zhukun Luo
Jiangxi university of finance and economics
'''
from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets('./mnist/',one_hot=True)
print("Training data size:",mnist.train.num_examples)
print("validating data size:",mnist.validation.num_examples)
print("testing data size:",mnist.test.num_examples)
print("Example training data:",len(mnist.train.images))#是个list

