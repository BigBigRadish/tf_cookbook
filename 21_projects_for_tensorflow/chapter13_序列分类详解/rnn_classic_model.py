# -*- coding: utf-8 -*-
'''
Created on 2018年11月22日

@author: Zhukun Luo
Jiangxi university of finance and economics
'''
import toySequenceData
from toySequenceData import ToySequenceData
import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
#运行的参数
learning_rate=0.01
training_iters=1000000
batch_size=128
display_step=10
#网络定义的参数
seq_max_len=20#最大的序列长度
n_hiddens=64#隐层的size
n_classes=2#类别数
trainset=ToySequenceData(n_samples=1000,max_seq_length=seq_max_len)
testset=ToySequenceData(n_samples=500,max_seq_length=seq_max_len)
#x为输入，y为输出
#None的位置实际为batch_size
x=tf.placeholder('float',[None,seq_max_len,1])
y=tf.placeholder('float',[None,n_classes])
seqlen=tf.placeholder(tf.int32,[None])#x的实际长度
#weights和bias在输出时被使用
weights={'out':tf.Variable(tf.random_normal([n_hiddens,n_classes]))}
bias={'out':tf.Variable(tf.random_normal([n_classes]))}
def dynamicRNN(x,seqlen,weights,bias):
    #输入x的形状(batch_size,seq_max_len,n_input)
    #输入seqlen的形状（batch-size,）
    #定义一个lstmcell,隐藏层大小为n_hidden
    lstm_cell=tf.nn.rnn_cell.BasicLSTMCell(n_hiddens)
    #使用tf.dynamic_rnn展开时间维度
    #此外sequence_length=seqlen他告诉每个序列应该运行多少步
    outputs,states=tf.nn.dynamic_rnn(lstm_cell,x,dtype=tf.float32,sequence_length=seqlen)
    batch_size=tf.shape(outputs)[0]
    index=tf.range(0,batch_size)*seq_max_len+(seqlen-1)
    outputs=tf.gather(tf.reshape(outputs,[-1,n_hiddens]),index)
    return tf.matmul(outputs,weights['out']+bias['out'])
pred=dynamicRNN(x, seqlen, weights, bias)#是logits,而不是概率，因此使用交叉熵定义损失
cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred,labels=y))#损失函数
optimizer=tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)#使用梯度下降
#分类准确率
correct_pred=tf.equal(tf.arg_max(pred,1),tf.arg_max(y,1))
accuracy=tf.reduce_mean(tf.cast(correct_pred,tf.float32))
#初始化
init=tf.global_variables_initializer()
#训练
config = tf.ConfigProto()  
config.gpu_options.allow_growth = True  
with tf.Session(config=config) as sess:
    sess.run(init)
    step=1
    while step*batch_size<training_iters:
        batch_x,batch_y,batch_seqlen=trainset.next(batch_size)
        sess.run(optimizer,feed_dict={x:batch_x,y:batch_y,seqlen:batch_seqlen})
        if step%display_step==0:
            acc=sess.run(accuracy,feed_dict={x:batch_x,y:batch_y,seqlen:batch_seqlen})
            loss=sess.run(cost,feed_dict={x:batch_x,y:batch_y,seqlen:batch_seqlen})
            print('iter'+str(step*batch_size)+",minibatch loss="+"{:6f}".format(loss)+",trainning accuracy="+"{:5f}".format(acc))
        step+=1
    print('optimizer finished!')
    test_data=testset.data
    test_label=testset.labels
    test_seqlen=testset.seqlen
    print('test accuracy:',sess.run(accuracy,feed_dict={x:test_data,y:test_label,seqlen:test_seqlen}))


    
