# -*- coding: utf-8 -*-
'''
Created on 2018年9月8日

@author: Zhukun Luo
Jiangxi university of finance and economics
'''
#利用tensorflow实现队列
import tensorflow as tf
#创建一个先进先出的队列，制定队列中最多保存两个元素，并指定类型为整数
q=tf.FIFOQueue(2,'int32')
#使用enqueue_many函数来初始化队列中的元素，和变量初始化类似，在使用队列之前需要明确调用这个初始化过程
init=q.enqueue_many(([0,10],), name='init')
#使用Dequeue函数将对垒中的第一个元素出队列，这个变量的值被存在变量x中
x=q.dequeue(name='init')
y=x+1#a tensor
q_inc=q.enqueue([y], name='init')
with tf.Session() as sess:
    init.run()
    for _ in range(5):
        #出队，+1，入队
        v,_=sess.run([x,q_inc])
        print(v,_)
#tensorflow 提供FIFOQueue()和RandomshuffleQueue()两种队列。第二种会将元素的顺序打乱，随机选择。队列不仅仅是一种数据结构，而且
#还是异步计算张量取值的一个机制。
#tensorfolw提供tf.coordinator和tf.QueueRunner两个类来完成多线程协同的功能。前者主要是用来协同多个线程同时停止。
import numpy as np
import threading
import time#
#线程中运行的程序，这个程序每隔1秒判断是否需要停止打印自己的ID
coord=tf.train.Coordinator()
def MyLoop(coord,work_id):
    #使用tf.Coordinator类提供的协同工具判断当前线程是否需要停止
    while not coord.should_stop():
        #随机停止所有线程
        if np.random.rand()<0.1:
            print('stoping from id: %d\n'% work_id,)
            coord.request_stop()#通知其他线程停止
        else:
            print('working on id: %d\n'% work_id,) 
            time.sleep(1)
threads=[threading.Thread(target=MyLoop,args=(coord,i)) for i  in range(5)]
#启动所有线程
for t in threads:
    t.start()
#waiting for thread exit
coord.join(threads)
