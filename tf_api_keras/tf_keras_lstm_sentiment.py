# -*- coding: utf-8 -*-
'''
Created on 2018年9月14日

@author: Zhukun Luo
Jiangxi university of finance and economics
'''
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense,Embedding
from keras.datasets import imdb
#最多使用的单词书
max_feature=20000
#循环神经网络的截断长度
maxlen=80
batch_size=32
#加载数据并将单词转化为ID，max_feature给出了最多使用的单词数。和自然
#语言模型类似，会将出现概率较低的单词替换为统一的Id。通过keras封装的api
#会生成25000条测试数据，每一条数据可以被看作一段话，并且每段话都有一个好评或者差评的标签
(trainX,trainY),(testX,testY)=imdb.load_data(num_words=max_feature)
print(len(trainX),'trainX')
#在自然语言处理中，。每一段话的长度是不一样的，但RNN的循环长度是固定的，所以这里需要先将所有段落统一成固定长度。
#对于长度不够的段落，使用0填充。对于超出的直接截断
trainX=sequence.pad_sequences(trainX,maxlen=maxlen)
testX=sequence.pad_sequences(testX,maxlen=maxlen)
'''
输出统一长度之后的数据
'''
print('x_train:',trainX.shape)
#在完成数据预处理之后构建模型
model=Sequential()
model.add(Embedding(max_feature,128))
#构建lstm
model.add(LSTM(128,dropout=0.2,recurrent_dropout=0.2))#只会得到最后一个节点的输出，如果输出每个时间点的结果,可以将return_sequences参数设置为True
model.add(Dense(1,activation='sigmoid'))
#与mnist样例类似地指定损失函数，优化函数与评测指标
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
#与mnist样例类似地指定训练数据，训练轮数，batch大小，以及验证数据
model.fit(trainX,trainY,batch_size=batch_size,epochs=15,validation_data=(testX,testY))
#在测试数据上评测模型
score=model.evaluate(testX, testY, batch_size=batch_size)
print('Test loss:',score[0])
print('test accuracy:',score[1])
