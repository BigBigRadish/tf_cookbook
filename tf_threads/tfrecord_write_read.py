# -*- coding: utf-8 -*-
'''
Created on 2018年9月8日

@author: Zhukun Luo
Jiangxi university of finance and economics
'''
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
#生成整数型的属性
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
#生成字符串型的属性
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
mnist=input_data.read_data_sets('/path/to/mnist/data',dtype=tf.uint8,one_hot=True)
images=mnist.train.images
#训练数据所对应的正确答案，可以作为一个属性保存在TFRECORD中
labels=mnist.train.labels
#训练数据的图像分辨率，可以作为一个Example中的一个属性
pixels=images.shape[1]
num_example=mnist.train.num_examples
#输出TFrecords文件的地址
filename='data/output.tfrecords'
#创建一个writer来写TFrecords文件
writer=tf.python_io.TFRecordWriter(filename)
for index in range(num_example):
    #将图像转化成一个字符串
    image_raw=images[index].toString()
    #将一个样例转化为Example Protocol Buffer，并将所有的信息写入这个数据结构
    example=tf.train.Example(features=tf.train.Features(feature={'pixels':_int64_feature(pixels),'label':_int64_feature(np.argmax(labels[index])),'image_raw':_bytes_feature(image_raw)}))
    #将一个Example写入TFRecord文件
    writer.write(example.SerializeTostring())
writer.close()
################################################################################################################################################################################################
#读入TFrecords文件
reader=tf.TFRecordReader()
#创建一个队列来维护输入文件列表 tf.train.string_input_producer
filename_queue=tf.train.string_input_producer(['data/output.tfrecords'])
#从文件中读出一个样例read_up_to。如果需要解析多个样例，可以用parse_example函数
_,serialized_example=reader.read(filename_queue)
features=tf.parse_single_example(serialized_example,features={'image_raw':tf.FixedLenFeature([],tf.string),'pixels':tf.FixedLenFeature([],tf.int64),'label':tf.FixedLenFeature([],tf.int64),})
#tf.decode_raw可以将字符串解析成图像对应的像素组
image=tf.decode_raw(features['image_raw'],tf.uint8)
label=tf.cast(features['label'],tf.int32)
pixels=tf.cast(features['pixels'],tf.int32)
sess=tf.Session()
#启动多线程处理输入数据
coord=tf.train.Coordinator()#多线程 同时停止
threads=tf.train.start_queue_runners(sess=sess,coord=coord)
for i in range(10):
    print(sess.run([image,label,pixels]))