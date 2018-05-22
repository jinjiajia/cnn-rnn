#!/usr/bin/python
# -*- coding: utf-8 -*-

from collections import Counter
from sklearn.model_selection import train_test_split
import tensorflow.contrib.keras as kr
import numpy as np
import os
import pandas as pd
import jieba

def wenti_read_file(filename):
    """读取文件数据"""
    data_xls=pd.read_excel(filename,usecols=[1,6],skiprows=[0])#skiprow=[0]去除第一行.usecols的第一列是B列，序号列不计入
    newdata=data_xls.values#to numpy
    alltext=newdata[:,0]
    alllab=np.array(newdata[:,1],np.int16)#dtype from object to int16
    lineswords=[]
    for i in range(len(alltext)):
        lineswords.append(' '.join(jieba.cut(alltext[i])))
    # trainlines,trainlab,testlines,testlab=train_test_split(alltext,alllab,test_size=0.1)
    # return trainlines,trainlab,testlines,testlab
    trainlines, testlines, trainlab, testlab = train_test_split(lineswords, alllab, test_size=0.1)
    return trainlines, testlines, trainlab, testlab


def build_vocab(data, vocab_size=5000):
    """根据训练集构建词汇表，存储"""
    # data, _ = wenti_read_file(filename)

    all_data = []
    for content in data:
        all_data.extend(content)

    counter = Counter(all_data)
    count_pairs = counter.most_common(vocab_size - 1)###提出最常用的
    words, _ = list(zip(*count_pairs))#zip返回两个tuple：一个是word的tuple，另一个是count的tuple
    # 添加一个 <PAD> 来将所有文本pad为同一长度
    words = ['<PAD>'] + list(words)

    open('data/cnews/vocab_cnews.txt', 'w',
        encoding='utf-8').write('\n'.join(words))

def read_vocab(filename):
    """读取词汇表"""
    words = list(map(lambda line: line.strip(),
        open(filename, 'r', encoding='utf-8').readlines()))
    word_to_id = dict(zip(words, range(len(words))))

    return words, word_to_id

def read_category():
    """读取分类目录，固定"""
    categories = ['婚姻家庭', '相邻纠纷', '人身赔偿', '涉老纠纷','涉少纠纷', '民间财产纠纷', '房屋纠纷', '消费纠纷',
                  '农村土地房屋', '交通事故纠纷','医疗纠纷','经营纠纷','金融纠纷','知识产权','劳动纠纷','物业纠纷']
    cat_to_id = dict(zip(categories, range(len(categories))))

    return categories, cat_to_id
# def _read_category():
#     """读取分类目录，固定"""
#     categories = ['体育', '财经', '房产', '家居',
#         '教育', '科技', '时尚', '时政', '游戏', '娱乐']
#     cat_to_id = dict(zip(categories, range(len(categories))))
#
#     return categories, cat_to_id

def to_words(content, words):
    """将id表示的内容转换为文字"""
    return ''.join(words[x] for x in content)

def _file_to_ids(contents,labels, word_to_id, max_length=600):
    """将文件转换为id表示"""
    # _, cat_to_id = read_category()
    # contents, labels = wenti_read_file(filename)

    data_id = []
    label_id = []
    # print(len(contents))
    # print(len(labels))
    for i in range(len(contents)):
        data_id.append([word_to_id[x] for x in contents[i] if x in word_to_id])
        label_id.append(labels[i])
        # label_id.append(cat_to_id[labels[i]])

    # 使用keras提供的pad_sequences来将文本pad为固定长度
    x_pad = kr.preprocessing.sequence.pad_sequences(data_id, max_length)
    y_pad = kr.utils.to_categorical(label_id)  # 将标签转换为one-hot表示
    # print(x_pad.shape)
    # print(y_pad.shape)
    # exit()
    return x_pad, y_pad

def preocess_file(trainlines,testlines,trainlab,testlab,seq_length=600):
    """一次性返回所有数据"""
    data_path='data/cnews'
    words, word_to_id = read_vocab(os.path.join(data_path,'vocab_cnews.txt'))
    # print(len(trainlines))
    # print(len(trainlab))
    x_train,y_train=_file_to_ids(trainlines,trainlab, word_to_id, seq_length)
    x_test,y_test=_file_to_ids(testlines,testlab, word_to_id, seq_length)

    # x_train, y_train = _file_to_ids(os.path.join(data_path,
    #     'cnews.train.txt'), word_to_id, seq_length)
    # x_test, y_test = _file_to_ids(os.path.join(data_path,
    #     'cnews.test.txt'), word_to_id, seq_length)
    # x_val, y_val = _file_to_ids(os.path.join(data_path,
    #     'cnews.val.txt'), word_to_id, seq_length)

    return x_train, y_train, x_test, y_test, words
data_path='data/cnews'
words, word_to_id = read_vocab(os.path.join(data_path,'vocab_cnews.txt'))
def preocess_line(lines,lab,seq_length=600):
    x_test,y_test=_file_to_ids(lines,lab, word_to_id, seq_length)
    return x_test,y_test
def batch_iter(data, batch_size=64, num_epochs=5):
    """生成批次数据"""
    data = np.array(data)
    data_size = len(data)
    num_batchs_per_epoch = int((data_size - 1) / batch_size) + 1
    for epoch in range(num_epochs):
        indices = np.random.permutation(np.arange(data_size))
        shuffled_data = data[indices]

        for batch_num in range(num_batchs_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


if __name__ == '__main__':
    trainlines,trainlab,testlines,testlab=wenti_read_file('wenti1.xlsx')
    if not os.path.exists('data/cnews/vocab_cnews.txt'):
         build_vocab(trainlines)
    x_train, y_train, x_test, y_test,words=preocess_file(trainlines,trainlab,testlines,testlab)


#     x_train, y_train, x_test, y_test, x_val, y_val = preocess_file()
#     print(x_train.shape, y_train.shape)
#     print(x_test.shape, y_test.shape)
#     print(x_val.shape, y_val.shape)
