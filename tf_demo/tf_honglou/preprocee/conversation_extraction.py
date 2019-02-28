# -*- coding: utf-8 -*-
'''
Created on 2019年2月28日

@author: Zhukun Luo
Jiangxi university of finance and economics
'''
import os
from tqdm import tqdm_notebook

def get_conversations(sentence):
    '''
        得到对话语句的起始位置
    Returns:
      start_index: int, 句子的开始位置
      end_index: int, 句子的结束索引
      conversation: str, 对话
    '''
    end_symbols = ['"', '“', '”']#结束标识符
    istart, iend = -1, -1
    talks = []
    ### get the start and end position for conversation
    for i in range(1, len(sentence)): 
        if (not istart == -1) and sentence[i] in end_symbols:
            iend = i
            conversation = {'istart':istart, 'iend':iend, 'talk':sentence[istart+1:iend]}
            talks.append(conversation)
            istart = -1
        if sentence[i-1] in [':', '：'] and sentence[i] in end_symbols:
            istart = i
    ### 从对话语句中提取对话人物
    contexts = []
    if len(talks):
        for i in range(len(talks)):
            if i == 0: 
                contexts.append(sentence[:talks[i]['istart']])
            else:
                contexts.append(sentence[talks[i-1]['iend']+1:talks[i]['istart']])
        # append the paragraph after the conversation if iend != len(sentence)
        if talks[-1]['iend'] != len(sentence):
            contexts.append(sentence[talks[-1]['iend']+1:])
        else:
            contexts.append(' ')
        ### the situation is not considered if the speaker comes after the talk
        for i in range(len(talks)):
            talks[i]['context'] = contexts[i] #+ 'XXXXX' + contexts[i+1]

    return talks, contexts            

def extract_corpus(book_name="../data/hongloumeng.txt", save_as="honglou.py"):
    fout = open(save_as, "w",encoding='UTF-8')
    with open(book_name, "r",encoding='UTF-8') as fin:
        fout.write('talks = [')
        for line in tqdm_notebook(fin.readlines()):
            talks, contexts = get_conversations(line.strip())
            if len(talks) > 0:
                for talk in talks: 
                    fout.write(talk.__repr__())
                    fout.write(',\n')
        fout.write(']')
    fout.close()

extract_corpus()

from honglou import talks

print(talks[-5:])

