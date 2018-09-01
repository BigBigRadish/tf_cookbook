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

        
        
 
if __name__ == '__main__':
    #filename=maybe_download('text8.zip', 31344016)
    filename='./text8.zip'
    vocabulary =read_data(filename)
    data,count,dictionary,reversed_dictionary=build_dataset(vocabulary,50000)
    del vocabulary
    print(data[:10]) 
