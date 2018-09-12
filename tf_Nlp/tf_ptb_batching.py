# -*- coding: utf-8 -*-
'''
Created on 2018年9月12日

@author: Zhukun Luo
Jiangxi university of finance and economics
'''
#在ptb数据集中每个句子并非随机抽取的文本，而是上下文有关联的内容。语言模型为了利用上下文信息，必须将前面句子的信息传递到后面的句子。为了实现这个目标，在ptb上下文关联的数据集中通常采用batching方法
TRAIN_DATA='dataset/data/ptb.train'#使用单词编号表示的数据集
TRAIN_BATCH_SIZE=20
TRAIN_NUM_STEP=35
import numpy as np
import csv
#从文件中读取数据，并返回表示三次编号的数组
def read_data(TRAIN_DATA):
    with open(TRAIN_DATA, mode='r') as fin:
        #将整个文档读进一个长字符串
        id_string=' '.join([line.strip() for line in fin.readlines()])
        id_list=[int(w) for w in id_string.split()]
        print(len(id_list))
        return id_list
def make_batch(id_list,batch_size,num_step):
    #计算总的batch数量。每个bauch保函的单词数量为batch_size*num_step
    num_batches=(len(id_list)-1)//(batch_size*num_step)
    #将数据整理成一个数组[num_batches,batch_size*num_step]
    data=np.array(id_list[:num_batches*batch_size*num_step])
    data=np.reshape(data, [batch_size,num_batches*num_step])
    data_batches=np.split(data,num_batches, axis=1)#沿着第二个维度将数据切分成num_batches个batch，存入一个数组
    #重复上述操作。但是每个位置向右移动一位，这里得到的时RNN每一步输出所需要预测的下一个单词
    label=np.array(id_list[1:num_batches*batch_size*num_step+1])
    label=np.reshape(data, [batch_size,num_batches*num_step])
    label_batchs=np.split(data,num_batches, axis=1)
    return list(zip(data_batches,label_batchs))
def main():
    train_batches=make_batch(read_data(TRAIN_DATA), TRAIN_BATCH_SIZE, TRAIN_NUM_STEP)
    print(train_batches)
    list2csv(train_batches, 'dataset/data/ptb.train.vector')
def list2csv(list, file):#将list写入文件
    wr = csv.writer(open(file, 'w'), quoting=csv.QUOTE_ALL)
    for word in list:
        wr.writerow([word])
if __name__ == '__main__':
    main()
    
