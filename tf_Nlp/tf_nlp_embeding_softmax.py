# -*- coding: utf-8 -*-
'''
Created on 2018年9月12日

@author: Zhukun Luo
Jiangxi university of finance and economics
'''
#基于循环神经网络的词向量层和softmax层
import tensorflow as tf
embedding=tf.get_variable('embedding',[VOCAB_SIZE,EMB_SIZE])#词汇表大小为VOCAB_SIZE，词向量维度为EMB_SIZE
#输出的矩阵比输入矩阵多一个维度，新增维度的大小是EMB_SIZE.在语言模型中，一般input_data的维度是batch_size*num_step,尔输出的input_embedding维度是batch_size*num_steps*EMB_SIZE
input_embedding=tf.nn.embedding_lookup(embedding,input_data)

#softmax层：是将RNN中的输出转化为一个单词表中的每个单词的输出概率
#Step1:使用一个线性映射将RNN的输出映射为一个维度与词汇表大小相同的向量。这一步的输出叫logits
#定义线性映射用到的参数
#hidden_size是循环神经网络的隐藏状态维度，VOCAB_SIZE是词汇表的大小
weight=tf.get_variable('weight',[HIDDEN_SIZE,VOCAB_SIZE])
bias=tf.get_variable('bias',[VOCAB_SIZE])
#计算线性映射
#计算线性映射
#output是RNN的输出，其维度为【batch_size*num_steps,hidden_size】
logits=tf.nn.bias_add(tf.matmul(output,weight),bias)
#step:
#调用softmax方法将logits转化为加和为1的概率。。语言模型的每一步输出都可以看作一个分类问题
#probs的维度与logits的维度相同
probs=tf.nn.softmax(logits)
#计算log Perplexity
loss=tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.reshape(self.targets,[-1]),logits=logits)
#通过共享参数减少参数数量

