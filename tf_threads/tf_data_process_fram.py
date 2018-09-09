# -*- coding: utf-8 -*-
'''
Created on 2018年9月8日

@author: Zhukun Luo
Jiangxi university of finance and economics
'''
import tensorflow as tf
files=tf.train.match_filenames_once('dataset/file-*')#得到所有的文件，并加入队列
filename_queue=tf.train.string_input_producer(files,shuffle=False)#创建一个输出队列
reader=tf.TFRecordReader()
_,serialized_example=reader.read(filename_queue)#从队列中读取文件
features=tf.parse_single_example(serialized_example,features={'image':tf.FixedLenFeature([],tf.string), 'label':tf.FixedLenFeature([],tf.int64),'height':tf.FixedLenFeature([],tf.int64),'width':tf.FixedLenFeature([],tf.int64),'channels':tf.FixedLenFeature([],tf.int64)})#定义数据的解析格式
image,label=features['image'],features['label']
height,width=features['height'],features['width']
channels=features['channels']
#从原始图像数据解析出像素矩阵，并根据图像尺寸还原图像
decode_image=tf.decode_raw(image,tf.uint8)#解析像素矩阵，并根据图像尺寸还原图像
decode_image.set_shape([height,width,channels])
#定义神经网络输入层图片的大小
image_size=299
distorted_image=None
#distorted_image=preprocess_for_train(decode_image,image_size,image_size,None)
min_after_dequeue=10000
batch_size=100
capacity=min_after_dequeue+3*batch_size
image_batch,label_batch=tf.train.shuffle_batch([distorted_image,label],batch_size=batch_size,capacity=capacity,min_after_dequeue=min_after_dequeue)
#定义神经网络结构及优化过程
learning_rate=0.001
logit=inference(image_batch)
loss=calc_loss(logit,label_batch)
train_step=tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
#声明会话并运行神经网络的优化过程
with tf.Session() as sess:
    sess.run((tf.global_variables_initializer(),tf.local_variables_initializer()))
    coord=tf.train.Coordinator()
    threads=tf.train.start_queue_runners(sess=sess,coord=coord)
    #神经网络训练过程
    TRAINING_ROUND=5000
    for i in range(TRAINING_ROUND):
        sess.run(train_step)
    #停止所有线程
    coord.request_stop()
    coord.join(threads)