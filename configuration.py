#!/usr/bin/python
# -*- coding: utf-8 -*-

class TCNNConfig(object):
    """CNN配置参数"""
    # 模型参数
    embedding_dim = 128      # 词向量维度
    seq_length = 600        # 序列长度
    num_classes = 16        # 类别数
    num_filters = 256       # 卷积核数目
    kernel_size = 3        # 卷积核尺寸
    vocab_size = 50000       # 词汇表达小

    hidden_dim = 128       # 全连接层神经元

    dropout_keep_prob = 0.5 # dropout保留比例训练0.5
    learning_rate = 1e-3   # 学习率
    istraining=1
    input_step=1
    batch_size = 128         # 每批训练大小
    num_epochs = 10        # 总迭代轮次

    print_per_batch = 100    # 每多少轮输出一次结果


class TRNNConfig(object):
    """RNN配置参数"""

    # 模型参数
    embedding_dim = 128      # 词向量维度
    seq_length = 600        # 序列长度
    num_classes = 10        # 类别数
    vocab_size = 50000       # 词汇表达小

    num_layers= 1           # 隐藏层层数
    hidden_dim = 128        # 隐藏层神经元
    rnn = 'lstm'             # lstm 或 gru
    istraining = 1  #positive is True
    dropout_keep_prob = 0.5 # dropout保留比例
    learning_rate = 1e-3    # 学习率

    batch_size = 64         # 每批训练大小
    num_epochs = 20          # 总迭代轮次
    print_per_batch = 100    # 每多少轮输出一次结果
