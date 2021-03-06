# -*- coding: utf-8 -*-
'''
Created on 2018年9月10日

@author: Zhukun Luo
Jiangxi university of finance and economics
'''
import tensorflow as tf
#列举输入文件。训练和测试使用不同的数据
train_files=tf.train.match_filenames_once('data/train_file-*')
test_files=tf.train.match_filenames_once('data/train_file_*')
#定义parser方法从TFRecord中解析数据。这里假设image中存储的是图像的原始向量
#label为该样例所对应的标签。height、width和channels给出了图片的维度
def parser(record):
    features=tf.parse_single_example(record,features={'image':tf.FixedLenFeature([],tf.string),'label':tf.FixedLenFeature([],tf.int64),'height':tf.FixedLenFeature([],tf.int64),'width':tf.FixedLenFeature([],tf.int64),'channels':tf.FixedLenFeature([],tf.int64)})
    #从原始图像中解析出像素矩阵，并根据图像尺寸还原图像
    decoded_image=tf.decode_raw(features['image'],tf.uint8)
    decoded_image.set_shape([features['height'],features['width'],features['channels']])
    label=features['label']
    return decoded_image,label
image_size=299 #定义神经网络输入层图片的大小
batch_size=100#定义组合数据batch的大小
shuffle_buffer=10000#定义随机打乱数据时buffer的大小

#定义读取训练数据的数据集
dataset=tf.data.TFRecordDataset(train_files)
dataset=dataset.map(parser)
#对数据依次进行预处理，shuffle和batching操作
dataset=dataset.map(lambda image,label:(preprocess_for_train(image,image_size,image_size,None),label))
dataset=dataset.shuffle(shuffle_buffer).batch(batch_size)
NUM_EPOCHS=10
dataset=dataset.repeat(NUM_EPOCHS)
#定义数据集迭代器。虽然定义数据集时没有直接使用placeholder来提供文件地址，但是
#tf.train.match_filenames_once方法得到的结果和与placeholder的机制类似
#也需要初始化，所以这里使用的时initializable_iterator
iterator=dataset.make_initializable_iterator()
image_batch,lable_batch=iterator.get_next()
#定义神经网络的结构及优化过程
learning_rate=0.001
logit=inference(image_batch)
loss=calc_loss(logit,lable_batch)
train_step=tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
#定义测试时用的Dataset