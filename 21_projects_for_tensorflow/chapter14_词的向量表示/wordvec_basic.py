# -*- coding: utf-8 -*-
'''
Created on 2018年11月22日

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
import math
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
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
#定义模型，一个单词预测另外一个单词，使用NCE损失
def build_model(vocabulary_size):
    batch_size=128
    embedding_size=128#wordvector词向量潜入空间位128维的向量
    skip_window=1
    num_skips=2
    valid_size=16#每次验证16个词
    valid_window=100#这16个词是从100个词中挑选出来的
    valid_examples=np.random.choice(valid_window,valid_size,replace=True)
    #构造损失时选取的噪声词数量
    num_example=64
    graphy=tf.Graph()
    with graphy.as_default():
        #输入的batch
        train_inputs=tf.placeholder(tf.int32,shape=[batch_size])
        train_labels=tf.placeholder(tf.int32,shape=[batch_size,1])
        #用于验证的词
        valid_dataset=tf.constant(valid_examples,dtype=tf.int32)
        with tf.device('/cpu:0'):
            embedding=tf.Variable(tf.random_uniform([vocabulary_size,embedding_size],-1.0,1.0))
            embd=tf.nn.embedding_lookup(embedding,train_inputs)
            #创建两个变量用于NCE loss
            nce_weights=tf.Variable(tf.truncated_normal([vocabulary_size,embedding_size],stddev=1.0/math.sqrt(embedding_size)))#
            nce_bias=tf.Variable(tf.zeros([vocabulary_size]))
            #tf.nn.nce_loss会自动选取噪声词，并且形成损失
            loss=tf.reduce_mean(tf.nn.nce_loss(weights=nce_weights,biases=nce_bias,labels=train_labels,inputs=embd,num_sampled=num_example,num_classes=vocabulary_size))
            optimizer=tf.train.GradientDescentOptimizer(1.0).minimize(loss)
            #对embedding层做归一化
            norm=tf.sqrt(tf.reduce_sum(tf.square(embedding),1,keep_dims=True))
            nomalized_embedding=embedding/norm
            #找出和验证词的embedding并计算他们和所有单词的相似度
            valid_embedding=tf.nn.embedding_lookup(nomalized_embedding,valid_dataset)
            similarity=tf.matmul(valid_embedding,nomalized_embedding,transpose_b=True)
            init=tf.global_variables_initializer()
            num_steps=100001
            with tf.Session(graphy=graphy) as session:
                #初始化变量
                init.run()
                print ('initialized !')
                average_loss=0
                for step in range(num_steps):
                    batch_inputs,batch_labels=generate_batch(batch_size, num_skips, skip_window)
                    feed_dict={train_inputs:batch_inputs,train_labels:batch_labels}
                    #优化一步
                    _,loss_val=session.run([optimizer,loss],feed_dict=feed_dict)#投食
                    average_loss=loss_val
                    if step%2000==0:#2000个batch的平均损失
                        if step>0:
                            print('Average LOSS at step',step,':',average_loss)
                    if step%10000==0:
                        #sim是验证词与词之间的相似度
                        sim=similarity.eval()
                        #一共有valid_size个验证词
                        for i in range(valid_size):
                            valid_word=reversed_dictionary[valid_examples[i]]
                            topk=8
                            nearest=(-sim[1,:]).argsort()[1:topk+1]
                            log_str='nearest to %s' %valid_word
                            for k in range(topk):
                                close_word=reversed_dictionary[nearest[k]]
                                log_str='%s %s,'%(log_str,close_word)
                            print(log_str)
                final_embed=nomalized_embedding.eval()
                return final_embed
def plot_with_labels(low_dim_embs,labels,filename='./tsn.png'):  
    assert low_dim_embs.shape[0]>len(labels),'more labels than embedding'
    plt.figure(figure=(8,18))#in inches
    for i ,label in enumerate(labels):
        x,y=low_dim_embs[1:]
        plt.scatter(x,y)
        plt.annotate(label,xy=(x,y),xytest=(5,2),textcoords='offset points',ha='right',va='bottom')
    plt.savefig(filename)
    
    
    
                                  
                            
                            
            
if __name__ == '__main__':
    vocabulary_size=50000
    #filename=maybe_download('text8.zip', 31344016)
    filename='./text8.zip'
    vocabulary =read_data(filename)
    data,count,dictionary,reversed_dictionary=build_dataset(vocabulary,50000)
    del vocabulary
#     print(data[:10])
#     data_index=0 
#     batch,labels=generate_batch(batch_size=8, num_skips=2, skip_window=1)
#     for i in range(8):
#         print(batch[i],reversed_dictionary[batch[i]],'->',labels[i,0],reversed_dictionary[labels[i,0]])
    final_embed=build_model(vocabulary_size)
    tsne=TSNE(perplexity=30,n_components=2,init='pca',n_iter=5000)
    plot_only=500
    low_dim_embs=tsne.fit_transform(final_embed[:plot_only,:])
    labels=[reversed_dictionary[i] for i in range(plot_only)]
    plot_with_labels(low_dim_embs, labels)