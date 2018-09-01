# -*- coding: utf-8 -*-
'''
Created on 2018年9月1日

@author: Zhukun Luo
Jiangxi university of finance and economics
'''
#生产序列数据
from __future__ import print_function
import tensorflow as tf
import random
import numpy as np
#这个类用于产生序列数据
class ToySequenceData(object):
    '''生成序列数据,每个序列可能具有不同的长度
                一共生成下面两类数据
      类别0:线性序列（如[0,1,2,3...]）
     类别1:完全随机序列（如[1,3,4,7...]）
    注意:max_seq_len是序列的最大长度，不足的填充为0，rnn中sequence_length属性进行序列长度的计算
    '''
    def __init__(self,n_samples=1000,max_seq_length=20,min_seq_length=3,\
                 max_value=1000):
        self.data=[]
        self.labels=[]
        self.seqlen=[]#用于存储序列长度
        for i in range(n_samples):
            len=random.randint(min_seq_length,max_seq_length)#长度随机
            self.seqlen.append(len)
            if random.random()<0.5:
                rand_start=random.randint(0,max_value-len)
                s=[[float(i)/max_value] for i in range(rand_start,rand_start+len)]#以0.5的概率生成一个线性序列
                s+=[[0.] for i in range(max_seq_length-len)]#长度不足后面补0
                self.data.append(s)
                self.labels.append([1.,0.])
            else:
                #生成一个随机序列
                s=[[float(random.randint(0,max_value))/max_value] for i in range(len)]
                s+=[[0.] for i in range(max_seq_length-len)]#长度不足后面补0
                self.data.append(s)
                self.labels.append([0.,1.])
        self.batch_id=0
    def next(self,batch_size):
        '''
        生成batch_size样本
        '''
        if self.batch_id==len(self.data):
            self.batch_id=0
        batch_data=(self.data[self.batch_id:min(self.batch_id+batch_size,len(self.data))])#数列数据
        batch_labels=(self.labels[self.batch_id:min(self.batch_id+batch_size,len(self.data))])#数列的标签
        batch_seqlen=(self.seqlen[self.batch_id:min(self.batch_id+batch_size,len(self.data))])#数列的真正长度
        self.batch_id=min(self.batch_id+batch_size,len(self.data))
        return batch_data,batch_labels,batch_seqlen
if __name__ == '__main__':
    SequenceData=ToySequenceData()
    batch_data,batch_labels,batch_seqlen=SequenceData.next(32)
    print(batch_data) 
            
                
            
        
    

