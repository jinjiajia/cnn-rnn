#!/usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf


class TextCNN(object):
    """文本分类，CNN模型"""
    def __init__(self, config):
        self.config = config

        self.input_x = tf.placeholder(tf.int32,
            [None, self.config.seq_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32,
            [None, self.config.num_classes], name='input_y')
        self.input_step = tf.placeholder(tf.int32,name='input_step')
        self.input_istrain=tf.placeholder(tf.int32,name='istraining')
        # self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        self.cnn()

    def input_embedding(self):
        """词嵌入，单词索引映射到低维的向量表示"""
        with tf.device('/cpu:0'):
            embedding = tf.get_variable('embedding',
                [self.config.vocab_size, self.config.embedding_dim])
            _inputs = tf.nn.embedding_lookup(embedding, self.input_x)
        return _inputs

    def cnn(self):
        """cnn模型"""
        embedding_inputs = self.input_embedding()

        with tf.name_scope("cnn"):
            # cnn 与全局最大池化
            conv = tf.layers.conv1d(embedding_inputs,
                self.config.num_filters,
                self.config.kernel_size, name='conv')

            # global max pooling
            gmp = tf.reduce_max(conv, reduction_indices=[1], name='gmp')

        with tf.name_scope("score"):
            # 全连接层，后面接dropout以及relu激活
            fc = tf.layers.dense(gmp, self.config.hidden_dim, name='fc1')
            # if self.input_istrain>0:
            #     fc = tf.contrib.layers.dropout(fc,self.config.dropout_keep_prob)
            # else:
            #     fc = tf.contrib.layers.dropout(fc,1.0)
            fc=tf.cond(self.input_istrain>0,lambda :tf.contrib.layers.dropout(fc,self.config.dropout_keep_prob),
                       lambda :tf.contrib.layers.dropout(fc,1.0))
            # fc = tf.cond(self.input_istrain > 0, lambda: tf.contrib.layers.dropout(fc, self.keep_prob),
            #              lambda: tf.contrib.layers.dropout(fc, 1.0))
            fc = tf.nn.relu(fc)

            # 分类器
            self.fc = fc
            self.logits = tf.layers.dense(fc, self.config.num_classes,
                                          name='fc2')
            #             self.pred_y = tf.nn.softmax(self.logits)
            self.y_pred_cls = tf.argmax(tf.nn.softmax(self.logits), 1)  # 预测类别

        with tf.name_scope("loss"):
            # 损失函数，交叉熵
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
                logits=self.logits, labels=self.input_y)
            self.loss = tf.reduce_mean(cross_entropy)

        with tf.name_scope("optimize"):
            # 优化器
            decay=pow(0.9,self.input_step/100)
            optimizer = tf.train.AdamOptimizer(
                learning_rate=self.config.learning_rate*decay)
            self.optim = optimizer.minimize(self.loss)

        with tf.name_scope("accuracy"):
            # 准确率
            correct_pred = tf.equal(tf.argmax(self.input_y, 1), self.y_pred_cls)
            self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
