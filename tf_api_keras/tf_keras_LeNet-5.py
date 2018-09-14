# -*- coding: utf-8 -*-
'''
Created on 2018年9月14日

@author: Zhukun Luo
Jiangxi university of finance and economics
'''
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense,Flatten,Conv2D,MaxPool2D
from keras import backend as k

num_classes=10
image_rows,image_cols=28,28
#通过keras封装好的api加载mnist数据。其中trainX就是毅一个600000*28*28的数组，trainY为每一张图片的数字
(trainX,trainY),(testX,testY)=mnist.load_data()
#因为不同的底层对输入的要求不一样，所以这里需要根据对图像编码的格式要求设置输入层的格式
if K.image_data_format()=='channels_first':
    trainX=trainX.reshape(trainX.shape[0],1,image_rows,image_cols)
    testX=testX.reshape(testX.shape[0],1,image_rows,image_cols)
    #因为MNIST中的图片是黑白的，所以第一维的取值为1
    input_shape=(1,image_rows,image_cols)
else:
    trainX=trainX.reshape(trainX.shape[0],image_rows,image_cols,1)
    testX=testX.reshape(testX.shape[0],image_rows,image_cols,1)
    #因为MNIST中的图片是黑白的，所以第一维的取值为1
    input_shape=(image_rows,image_cols,1)
#将图像转为0到1之间的实数
trainX=trainX.astype('float32')
testX=testX.astype('float32')
trainX/=255.0
testX/255.0
#将标准答案转化为需要的格式(one-hot编码)
trainY=keras.utils.to_categorical(trainY,num_classes)
testY=keras.utils.to_categorical(testY,num_classes)
#使用Keras API定义模型
model=Sequential()
#一层深度为32，过滤器大小为5*5的卷积层
model.add(Conv2D(32,kernel_size=(5,5),activation='relu',input_shape=input_shape)) 
#一层过滤器大小为2*2的最大池化层
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Conv2D(64,(5,5),activation='relu'))  
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(500,activation='relu')) 
model.add(Dense(num_classes,activation='softmax'))
#定义损失函数
model.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.SGD(),metrics=['accuracy'])
#类似TFlearn中的训练过程，给出训练数据，batch的大小，训练轮数和验证数据，keras可以自动完成模型训练过程
model.fit(trainX,trainY,batch_size=128,epochs=20,validation_data=(testX,testY))
#在测试数据上计算准确率
score=model.evaluate(testX,testY)
print('Test loss:',score[0])
print('test accuracy;',score[1])
    