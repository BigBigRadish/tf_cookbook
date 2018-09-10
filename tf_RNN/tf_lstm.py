# -*- coding: utf-8 -*-
'''
Created on 2018年9月10日

@author: Zhukun Luo
Jiangxi university of finance and economics
'''
from tensorflow.contrib.layers.python.layers.layers import fully_connected
'''
循环神经网络通过保存历史信息来帮助当前的决策
lstm主要用来解决长期依赖问题
与单一的tanh循环体结构不同，lstm拥有三个‘门’结构
‘门的结构’：使用sigmoid神经网络和一个按位乘法的操作sigmoid（0，1），相当于信息的门
为了使RNN更有效保存长期记忆。‘遗忘门’和‘输入门’就至关重要
‘遗忘门’：f=sigmoid(W1x+W2h)
当RNN忘记了部分之前的状态后，他还需要从当前的输入补充最新的记忆，这个过程就是输入门完成的。
'''
#定义一个LSTM结构。在tensorflow中通过一句简单的命令就可以实现一个完整的LSTM结构
#lstm中使用的变量也会在函数中自动被声明
import tensorflow as tf
lstm=tf.nn.rnn_cell.BasicLSTMCell(lstm_hidden_size)
#将lstm中的状态初始化为全0的数组。BasicLSTMCell类提供了zero_state函数来生成全0的初始状态。state是一个包含两个张量的lstmstatetuple类，其中state。c和state。h分别对应了c状态和h状态
state=lstm.zero_state(batch_size, dtype=tf.float32)
#定义损失函数
loss=0.0
#虽然在测试RNN可以处理任意长度的序列，但是在训练中为了将循环网络展开成前馈网络，我们需要知道训练数据的序列长度。用num_step表示其长度
#dynamic_rnn是动态处理变长的方法
for i in range(num_steps):
    #在第一个时刻声明lstm结构中使用的变量，在之后的时刻都需要复用之前定义好的变量
    if i>0:
        tf.get_variable_scope().reuse_variables()
    #每一步处理时间序列中的一个时刻，将当前输入current_input和前一时刻状态state（h1和从）传入定义的lstm结构可以得到当前的lstm的输出lstm_output和更新后状态state
    lstm_output,state=lstm(current_input,state)
    final_output=fully_connected(lstm_output)
    loss+=calc_loss(final_output,expect_output)