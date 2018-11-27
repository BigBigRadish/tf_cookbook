# -*- coding: utf-8 -*-
'''
Created on 2018年9月1日

@author: Zhukun Luo
Jiangxi university of finance and economics
'''
#自回归模型
from __future__ import print_function
import numpy as np
import matplotlib 
from tensorflow.contrib.timeseries.examples.lstm import _LSTMModel
matplotlib.use('agg')
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.contrib.timeseries.python.timeseries import estimators as ts_estimators
from tensorflow.contrib.timeseries.python.timeseries import NumpyReader,CSVReader
#接着，利用np.sin生成一个实验用的时间序列数据，改时间序列数据实际上是在正弦曲线上加入了上升的趋势和一些随机的噪声
x=np.array(range(1000))
noise=np.random.uniform(-0.2,0.2,1000)
y=np.sin(np.pi*x/50)+np.cos(np.pi*x/50)+np.sin(np.pi*x/25)+noise
data={#以numpy的形式读入
    tf.contrib.timeseries.TrainEvalFeatures.TIMES:x,
    tf.contrib.timeseries.TrainEvalFeatures.VALUES:y
    }
reader=NumpyReader(data)
#建立batch数据集
train_input_fn=tf.contrib.timeseries.RandomWindowInputFn(reader,batch_size=4,window_size=100)
#隐层为128
estimator=ts_estimators.TimeSeriesRegressor(model=_LSTMModel(num_features=1,num_units=128),optimizer=tf.train.AdadeltaOptimizer(0.001))
estimator.train(input_fn=train_input_fn, steps=2000)
evaluation_input_fn=tf.contrib.timeseries.WholeDatasetInputFn(reader)
evaluation=estimator.evaluate(input_fn=evaluation_input_fn, steps=1)
(predictions,)=tuple(estimator.predict(input_fn=tf.contrib.timeseries.predict_continuation_input_fn(evaluation,steps=200)))
observed_times=evaluation['times'][0]
observed=evaluation['observed'][0,:,:]
evaluated_times=evaluation['times'][0]
evaluated=evaluation['mean'][0]
predicted_times=predictions['times']
predicted=predictions['mean']
plt.figure(figsize=(15,5))
plt.axvline(999, linestyle='dotted',linewidth=4,color='r')
observed_lines=plt.plot(observed_times,observed,label='observation',color='k')
evaluated_lines=plt.plot(evaluated_times,evaluated,label='evaluation',color='g')
predicted_lines=plt.plot(predicted_times,predicted,label='prediction',color='r')
plt.legend(handles=[observed_lines[0],evaluated_lines[0],predicted_lines[0]],loc='upper left')
plt.savefig('./picture_result1.jpg')
plt.show()

