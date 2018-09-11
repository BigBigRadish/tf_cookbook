# -*- coding: utf-8 -*-
'''
Created on 2018年9月11日

@author: Zhukun Luo
Jiangxi university of finance and economics
'''
import tensorflow as tf
#假设词汇表的大小是3，语料包含两个单词
word_labels=tf.constant([2,0])
#假设模型对两个单词预测时，产生的logit分别是【2.0，-1.0，3.0】和[1.0,0.0,-0.5]
#注意这里的logit不是概率，因此他们不是0.0-1.0范围之间的数字，如果需要计算概率，则需要调用prob=tf.nn.softmax(logits).这里交叉熵直接使用logit即可
predict_logits=tf.constant([[2.0,-1.0,3.0],[1.0,0.0,-0.5]])
#使用sparse_sofmax_cross_entropy_with_logits计算交叉熵
loss=tf.nn.sparse_softmax_cross_entropy_with_logits(labels=word_labels, logits=predict_logits)
#运行程序，计算loss的结果，对应于两个预测的perplecity损失
sess=tf.Session()
print(sess.run(loss))
#softmax_cross_entropy_with_logits与上面的函数相似，但是需要将预测慕白哦以概率分布的形式给出
word_prob_distribution=tf.constant([[0.0,0.0,1.0],[1.0,0.0,0.0]])
loss=tf.nn.softmax_cross_entropy_with_logits(labels=word_prob_distribution, logits=predict_logits)
print(sess.run(loss))
#由于soft_max_cross_entropy_with_logits允许提供一个概率分布，因此在使用时有更大的自由度。
#举个例子，一种叫做label smoothing的技巧是将正确数据的概率设为一个比1.0略小的值，将错误数据设为比0.0略大的值，这样可以避免模型与数据overfitting,在某些时候可以提高训练效果
word_prob_smooth=tf.constant([[0.01,0.01,0.98],[0.98,0.01,0.01]])
loss=tf.nn.softmax_cross_entropy_with_logits(labels=word_prob_smooth,logits=predict_logits)
print(sess.run(loss))
