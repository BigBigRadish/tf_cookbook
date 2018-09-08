# -*- coding: utf-8 -*-
'''
Created on 2018年9月8日

@author: Zhukun Luo
Jiangxi university of finance and economics
'''
#tf.Coordinator 和 tf。QueueRunner管理多线程队列实例
import tensorflow as tf
#声明一个先进先出的队列，队列中最多100个元素，且全为实数
queue=tf.FIFOQueue(100,'float')
#定义队列的入队操作
enqueue_op=queue.enqueue([tf.random_normal([1])])
#使用tf.train.QueueRunner来创建多个线程运行队列的入队操作
#tf.train.QueueRunner的第一个参数给出了被操作的队列，【enqueue_op】*5表示需要启动5个线程，每个线程中运行的enqueue_op操作
qr=tf.train.QueueRunner(queue,[enqueue_op]*5)
#将定义过的QueueRunner加入tensorflow计算图指定的集合
#tf.train.add_queue_runner函数没有制定集合，则默认加入集合tf.GraphKeys.queue_runners..
#将qr加入默认的图集合
tf.train.add_queue_runner(qr)
#定义出队操作
out_tensor=queue.dequeue()
with tf.Session() as sess:
    #使用tf.train.coordinator来协同启动线程
    coord=tf.train.Coordinator()
    #使用tf。train.QueueRunner时，需要明确调用tf.train.start_queue_runners来启动所有线程。否则因为没有线程运行入队操作
    threads=tf.train.start_queue_runners(sess,coord)
    for _ in range(3):
        print (sess.run(out_tensor)[0])
    coord.request_stop()
    coord.join(threads)
        