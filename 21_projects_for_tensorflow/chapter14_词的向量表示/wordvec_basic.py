# -*- coding: utf-8 -*-
'''
Created on 2018年9月1日

@author: Zhukun Luo
Jiangxi university of finance and economics
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import os
import random
import zipfile
import numpy as np
import urllib
import tensorflow as tf
url='http://mattmahoney.net/dc/'
def maybe_download(filename,expected_bytes):#数据下载
    if not os.path.exists(filename):
        filename,_=urllib.request.urlretrieve(url+filename, filename)
    statinfo=os.stat(filename)
    if statinfo.st_size==expected_bytes:
        print('found and writed',filename)
    else:
        print(statinfo.st_size)
    raise Exception('failed to verify'+filename+'can u get to it with a browser?')
#将语料库解压
def read_data(filename):
    '''
    将下载好的zip文件进行解压并读取为word的list
    '''
    with zipfile.ZipFile(filename) as f:
        data=tf.compat.as_str(f.read(f.namelist()[0])).split()
        return data
#制作此表并对之前的语料库进行转换
#其主要步骤：1.制作一个此表，将一个单词映射为一个ID 2.词表的大小设置为5w(即只考虑最常用的词)3.将不常见的词映射为unk标识符
def build_dataset(words,n_words):
    '''
    将原始的单词表映射为单词
    '''
    count=[['unk',-1]]
    count.extend(collections.Counter(words).most_common(n_words-1))#以列表的形式统计最常用的词的频率
    dictionary=dict()
    for word,_ in count:
        dictionary[word]=len(dictionary)#逐渐递增
    data=list()
    unk_count=0
    for word in words:
        if word in dictionary:
            index=dictionary[word]
        else:
            index=0
            unk_count+=1
        data.append(index)#替换
    count[0][1]=unk_count
    reversed_dictionary=dict(zip(dictionary.values(),dictionary.keys()))
    return data,count,dictionary,reversed_dictionary#用数字代替的数据，对应词的频率，每个词对应的id
#生成每一步的训练样本，使用skip-gram
def generate_batch(batch_size,num_skips,skip_window):#产生批数据处理函数
    
    # data_index相当于一个指针，初始为0
    #每次生成一个batch，data_index会相应后撤
    global data_index
    assert batch_size%num_skips==0
    assert num_skips<=2*skip_window
    batch=np.ndarray(shape=(batch_size),dtype=np.int32)
    labels=np.ndarray(shape=(batch_size,1),dtype=np.int32)
    span=2*skip_window+1#[skip_window target skip_window]
    buffer=collections.deque(maxlen=span)
    #data_index是当前数据开始的位置
    #产生batch后往后推1位
    for _ in range(span):
        buffer.append(data[data_index])
        data_index=(data_index+1)%len(data)
    for i in range(batch_size//num_skips):
        #利用buffer生成batch
        #buffer是一个长度位、为2*skip_window+1长度的word list
        #一个buffer生成num_skips个数的样本
        target=skip_window#target label at the center of the buffer
        target_to_avoid=[skip_window]#保证样本不重复
        for j in range(num_skips):
            while target in target_to_avoid:
                target=random.randint(0,span-1)
            target_to_avoid.append(target)
            batch[i*num_skips+j]=buffer[skip_window]
            labels[i*num_skips+j,0]=buffer[target]
        buffer.append(data[data_index])
    data_index=(data_index+len(data)-span)%len(data)
    return batch,labels
    #默认情况下skip-window=1,num_skip=2
    #此时是从连续的3个词钟生成2个样本
        
        
        

        
        
 
if __name__ == '__main__':
    #filename=maybe_download('text8.zip', 31344016)
    filename='./text8.zip'
    vocabulary =read_data(filename)
    data,count,dictionary,reversed_dictionary=build_dataset(vocabulary,50000)
    del vocabulary
#     print(data[:10])
    data_index=0 
    batch,labels=generate_batch(batch_size=8, num_skips=2, skip_window=1)
    for i in range(8):
        print(batch[i],reversed_dictionary[batch[i]],'->',labels[i,0],reversed_dictionary[labels[i,0]])
