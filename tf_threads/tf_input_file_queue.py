# -*- coding: utf-8 -*-
'''
Created on 2018年9月9日

@author: Zhukun Luo
Jiangxi university of finance and economics
'''
#虽然一个tfrecords文件中可以存储多个训练样例，但是当询量数据量较大时，可以将数据分成多个tfrecord文件来提高处理效率。tensorflow提供了
#tf.train.match_filenames_once函数来获取一个符合正则表达式的所有文件，得到的文件列表可以通过tf.train.string_input_producer函数进行有效的管理
#tf.train.string_input_producer函数会使用初始化时提供的文件列表创建一个输入队列，输入队列中原始的元素为文件列表中的所有文件，输入队列可以作为文件读取函数的参数。
#每次调用文件读取函数时，该函数会先判断当前是否已有打开的文件可读，如果没有或者打开的文件已经读完，这个函数会从输入队列出对一个文件读取数据
#当一个输入队列的所有文件都被处理完，又会重新初始化将文件全部加载进队列，可以通过设置num_epochs参数限制加载的轮数，一般测试数据设定num_epochs=1,加载一次就行
import tensorflow as tf
#创建TFRecord 文件的帮助函数
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
#模拟海量数据情况下将数据写入不同的文件。num_shards定义了总共写入多少个文件
#instance_per_shard定义了每个文件有多少个数据
num_shards=2#定义多少个文件
instances_per_shards=2#定义每个文件有多少个数据
for i in range(num_shards):
    filename=('./dataset/data.tfrecords-%.5d-of-%.5d'%(i,num_shards))#这里需要手动创建文件夹，否则会报错tensorflow.python.framework.errors_impl.NotFoundError: Failed to create a NewWriteableFile
    writer=tf.python_io.TFRecordWriter(filename)
    #将数据封装成Example结构并写入TFRecord文件
    for j in range(instances_per_shards):
        example=tf.train.Example(features=tf.train.Features(feature={'i':_int64_feature(i),'j':_int64_feature(j)}))
        writer.write(example.SerializeToString())
    writer.close()
##############################################################################################################################################################################################
#读取tfRECORD文件
files=tf.train.match_filenames_once('./dataset/data.tfrecords-*')#使用tf.train.match_filenames_once函数获取文件列表
filename_queue=tf.train.string_input_producer(files,shuffle=False)#在真实环境中一般时设置shuffle为True，打乱顺序
reader=tf.TFRecordReader()
_,serialized_example=reader.read(filename_queue)
features=tf.parse_single_example(serialized=serialized_example,features={'i':tf.FixedLenFeature([],tf.int64),'j':tf.FixedLenFeature([],tf.int64)})
with tf.Session() as sess:
    tf.local_variables_initializer().run()#使用tf.train.match_filenames_once()时需要先初始化
    print(sess.run(files))
    coord=tf.train.Coordinator()
    threads=tf.train.start_queue_runners(sess=sess,coord=coord)
    #多次执行获取数据的操作
    for i in range(6):
        print(sess.run([features['i'],features[
            'j']]))
    coord.request_stop()
    coord.join(threads)