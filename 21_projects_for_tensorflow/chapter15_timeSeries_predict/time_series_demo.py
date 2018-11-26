# -*- coding: utf-8 -*-
'''
Created on 2018年11月26日

@author: Zhukun Luo
Jiangxi university of finance and economics
'''
from __future__ import print_function
import numpy as np
import matplotlib 
matplotlib.use('agg')
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.contrib.timeseries.python.timeseries import NumpyReader
#接着，利用np.sin生成一个实验用的时间序列数据，改时间序列数据实际上是在正弦曲线上加入了上升的趋势和一些随机的噪声
x=np.array(range(1000))
noise=np.random.uniform(-0.2,0.2,1000)
y=np.sin(np.pi*x/100)+x/200+noise
plt.plot(x,y)
plt.savefig('timeseries_y.jpg')
data={#以numpy的形式读入
    tf.contrib.timeseries.TrainEvalFeatures.TIMES:x,
    tf.contrib.timeseries.TrainEvalFeatures.VALUES:y
    }
reader=NumpyReader(data)
with tf.Session() as sess:
    full_data=reader.read_full()#返回时间序列对应的tensor
    #调用read_full()方法会生成读取队列
    #调用tf.trian.start_queue_runners启动队列才能正常读取
    coord=tf.train.Coordinator()
    threads=tf.train.start_queue_runners(sess=sess, coord=coord)
    print(sess.run(full_data))
    coord.request_stop()
#建立batch数据集
train_input_fn=tf.contrib.timeseries.RandomWindowInputFn(reader,batch_size=2,window_size=10)
with tf.Session() as sess:
    batch_data=train_input_fn.create_batch()
    coord=tf.train.Coordinator()
    threads=tf.train.start_queue_runners(sess=sess, coord=coord)
    one_batch=sess.run(batch_data[0])
    coord.request_stop()
print(one_batch)