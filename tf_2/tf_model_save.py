# -*- coding: utf-8 -*-
'''
Created on 2018年9月4日

@author: Zhukun Luo
Jiangxi university of finance and economics
'''
import tensorflow as tf
#声名两个变量并计算它们的和
# v1=tf.Variable(tf.constant(1.0,shape=[1]),name='v1')
# v2=tf.Variable(tf.constant(2.0,shape=[1]),name='v2')
# result=v1+v2
# init_op=tf.global_variables_initializer()
#声明tf.train.Saver类用于保存模型
#saver=tf.train.Saver()
saver=tf.train.import_meta_graph("./model/model.ckpt.meta")
with tf.Session() as sess:
   # sess.run(init_op)
    #saver.save(sess, './model/model.ckpt')

#model.ckpt.meta,保存计算图的结构

#     saver.restore(sess,'./model/model.ckpt')#加载模型进行计算
#     print(sess.run(result))
#直接加载模型
    saver.restore(sess,'./model/model.ckpt')#加载模型进行计算
    print(sess.run(tf.get_default_graph().get_tensor_by_name("add:0")))#可以将前几层存储为模型，后一层进行训练
            #模型加载的时候阔以给变量进行重命名
    
    
    
