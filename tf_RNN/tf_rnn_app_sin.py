# -*- coding: utf-8 -*-
'''
Created on 2018年9月10日

@author: Zhukun Luo
Jiangxi university of finance and economics
'''
import numpy as np
import tensorflow as tf
 
import matplotlib as mpl
mpl.use('TkAgg')
from matplotlib import pyplot as plt
'''
HIDDEN_SIZE表示隐藏节点个数，NUM_LAYERS表示LSTM的层数，循环神经网络序列长度为10，训练步数10000次，batch大小32；同样的为了检测训练的准确性，设置10000训练数据个数，和1000个测试数据个数，并且采样间隔设置为0.01。
'''
HIDDEN_SIZE = 30
NUM_LAYERS = 2
 
TIMESTEPS = 10
TRAINING_STEPS = 10000
BATCH_SIZE = 32
 
TRAINING_EXAMPLES = 10000
TESTING_EXAMPLES = 1000
SAMPLE_GAP = 0.01
'''
generate_data函数用于生成数据，采用序列长为标准，取从i到i+TIMESTEPS-1的长度作为输入，预测第i+TIMESTEPS的输出，分别反会训练X和标签y。
'''
def generate_data(seq):
    X = []
    y = []
    for i in range(len(seq)-TIMESTEPS):
        X.append([seq[i:i+TIMESTEPS]])
        y.append(seq[i+TIMESTEPS])
    return np.array(X,dtype=np.float32),np.array(y,dtype=np.float32)
''''
lstm_model函数运用多层LSTM结构，本例采用2层，通过函数tf.nn.rnn_cell.BasicLSTMCell创建单个cell，再用tf.nn.rnn_cell.MultiRNNCell函数合成多层的LSTM结构，
通过tf.nn.dynamic_rnn函数就可以运行，其中cell为神经网络结构，X为输入的序列参数，dtype类型取float32。
'''
def lstm_model(X,y,is_training):
    cell = tf.nn.rnn_cell.MultiRNNCell([
        tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE)
        for _ in range(NUM_LAYERS)
    ])
 
    outputs,_ = tf.nn.dynamic_rnn(cell,X,dtype=tf.float32)
    output = outputs[:,-1,:]
    predictions = tf.contrib.layers.fully_connected(output,1,activation_fn=None)
 
    if not is_training:
        return predictions,None,None
 
    loss = tf.losses.mean_squared_error(labels=y,predictions=tf.reshape(predictions,[-1]))
 
    train_op = tf.contrib.layers.optimize_loss(
        loss,tf.train.get_global_step(),
        optimizer = "Adagrad",learning_rate = 0.1
    )
 
    return predictions,loss,train_op
'''
train函数训练，计算loss值并每隔100次输出；再利用run_eval函数测试性能，计算均方根误差rmse输出，之后分别绘制预测的正弦函数曲线和真实正弦函数曲线，作对比
'''
def train(sess,train_X,train_y):
    ds=tf.data.Dataset.from_tensor_slices((train_X,train_y))
    ds = ds.repeat().shuffle(1000).batch(BATCH_SIZE)
    X,y = ds.make_one_shot_iterator().get_next()
 
    with tf.variable_scope("model"):
        predictions,loss,train_op = lstm_model(X,y,True)
 
    sess.run(tf.global_variables_initializer())
    for i in range(TRAINING_STEPS):
        _,l = sess.run([train_op,loss])
        if i % 100 == 0:
            print("train steps; " + str(i) + ', loss; ' + str(l))
 
def run_eval(sess,test_X,test_y):
    ds = tf.data.Dataset.from_tensor_slices((test_X,test_y))
    ds = ds.batch(1)
    X,y = ds.make_one_shot_iterator().get_next()
 
    with tf.variable_scope("model",reuse=True):
        prediction,_,_ = lstm_model(X,[0.0],False)
 
    predictions = []
    labels = []
    for i in range(TESTING_EXAMPLES):
        p,l = sess.run([prediction,y])
        predictions.append(p)
        labels.append(l)
 
    predictions = np.array(predictions).squeeze()
    labels = np.array(labels).squeeze()
    rmse = np.sqrt(((predictions-labels)**2).mean(axis=0))
    print ("Mean Square Error is: %f" % rmse)
 
    plt.figure()
    plt.plot(labels,label='real_sin',linewidth=2)
    plt.plot(predictions,label="predictions",linewidth=1)
    plt.legend()
    plt.show()

test_start = (TRAINING_EXAMPLES+TIMESTEPS)*SAMPLE_GAP
test_end = test_start+(TESTING_EXAMPLES+TIMESTEPS)*SAMPLE_GAP
train_X,train_y = generate_data(np.sin(np.linspace(0,test_start,TRAINING_EXAMPLES+TIMESTEPS,dtype=np.float32)))
test_X,test_y = generate_data(np.sin(np.linspace(test_start,test_end,TESTING_EXAMPLES+TIMESTEPS,dtype=np.float32)))
 
with tf.Session() as sess:
    train(sess,train_X,train_y)
    run_eval(sess,test_X,test_y)