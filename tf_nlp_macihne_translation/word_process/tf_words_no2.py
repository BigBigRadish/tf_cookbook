# -*- coding: utf-8 -*-
'''
Created on 2018年9月12日

@author: Zhukun Luo
Jiangxi university of finance and economics
'''
#确定词汇表之后，再将训练文件，测试文件等根据词汇文件转化为单词编号。每个单词的编号就是它在词汇文件中的行号
import codecs
import sys
RAW_DATA='../dataset/train.txt.zh'#训练集数据文件
VOCAB='../dataset/train.zh.vocab'#输出的词汇表文件
OUTPUT_DATA='../dataset/ptb.train.zh'#将单词替换成单词编号后的输出

 
# 读取词汇表，并建立词汇到单词编号的映射。
with codecs.open(VOCAB, "r", "utf-8") as f_vocab:
    vocab = [w.strip() for w in f_vocab.readlines()]
word_to_id = {k: v for (k, v) in zip(vocab, range(len(vocab)))}
 
# 如果出现了不在词汇表内的低频词，则替换为"unk"。
 
 
def get_id(word):
    return word_to_id[word] if word in word_to_id else word_to_id["<unk>"]
 
 
fin = codecs.open(RAW_DATA, "r", "utf-8")
fout = codecs.open(OUTPUT_DATA, 'w', 'utf-8')
for line in fin:
    words = line.strip().split() + ["<eos>"]  # 读取单词并添加<eos>结束符
    # 将每个单词替换为词汇表中的编号
    out_line = ' '.join([str(get_id(w)) for w in words]) + '\n'
    fout.write(out_line)
fin.close()
fout.close()
