'''
Created on 2018年9月9日

@author: Zhukun Luo
Jiangxi university of finance and economics
'''
#组合训练数据，将多个输入样例组合成一个batch
#TENSORFLOW提供tf.train.batch和tf。train。shuffle_batch来组织单个样例
#队列的入队是生成单个样例的方法，而出队是生成batch的方法
import tensorflow as tf
from tf_input_file_queue import features
example,label=features['i'],features['j']
#一个batch中的样例个数
batch_size=3
capacity=1000+3*batch_size#设置队列的大小
example_batch,label_batch=tf.train.batch([example,label],batch_size=batch_size,capacity=capacity)
with tf.Session() as sess:
    tf.initialize_all_variables().run()
    coord=tf.train.Coordinator()
    threads=tf.train.start_queue_runners(sess=sess,coord=coord)
    #获取并打印组合的样例，在真实问题中，这个输出一般作为神经网络的输入
    for i in range(2):
        cur_example_batch,cur_label_batch=sess.run([example_batch,label_batch])
        print(cur_example_batch,cur_label_batch)
    coord.request_stop()
    coord.join(threads)