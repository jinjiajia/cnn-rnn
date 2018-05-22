#!/usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf


class TextRNN(object):
    """文本分类，RNN模型"""
    def __init__(self, config):
        self.config = config

        self.input_x = tf.placeholder(tf.int32,
            [None, self.config.seq_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32,
            [None, self.config.num_classes], name='input_y')
        self.input_step = tf.placeholder(tf.int32,name='input_step')
        self.input_istrain=tf.placeholder(tf.bool,name='istraining')

        self.rnn()

    def input_embedding(self):
        """词嵌入"""
        with tf.device('/cpu:0'):
            embedding = tf.get_variable('embedding',
                [self.config.vocab_size, self.config.embedding_dim])
            _inputs = tf.nn.embedding_lookup(embedding, self.input_x)
        return _inputs

    def rnn(self):
        """rnn模型"""

        def lstm_cell():
            """lstm核"""
            return tf.contrib.rnn.BasicLSTMCell(self.config.hidden_dim,
                state_is_tuple=True)

        def gru_cell():
            """gru核"""
            return tf.contrib.rnn.GRUCell(self.config.hidden_dim)

        def dropout():
            """为每一个rnn核后面加一个dropout层"""
            if (self.config.rnn == 'lstm'):
                cell = lstm_cell()
            else:
                cell = gru_cell()
            if self.input_istrain>0:
                return tf.contrib.rnn.DropoutWrapper(cell,
                output_keep_prob=self.config.dropout_keep_prob)
            else:
                return tf.contrib.rnn.DropoutWrapper(cell,
                output_keep_prob=1.0)

        embedding_inputs = self.input_embedding()

        with tf.name_scope("rnn"):
            # 多层rnn网络
            cells = [dropout() for _ in range(self.config.num_layers)]
            rnn_cell = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)
            if(self.config.rnn=='bilstm'):
                cells_bw=[dropout() for _ in range(self.config.num_layers)]
                rnn_cell_bw = tf.contrib.rnn.MultiRNNCell(cells_bw, state_is_tuple=True)
                # _outputs,_,_= rnn.bidirectional_rnn(rnn_cell, rnn_cell_bw,inputs=embedding_inputs, dtype=tf.float32)
                _outputs,_=tf.nn.bidirectional_dynamic_rnn(rnn_cell, rnn_cell_bw,inputs=embedding_inputs, dtype=tf.float32)
            else:
                _outputs, _ = tf.nn.dynamic_rnn(cell=rnn_cell,inputs=embedding_inputs, dtype=tf.float32)
            last = _outputs[:, -1, :]  # 取最后一个时序输出作为结果

        with tf.name_scope("score"):
            # 全连接层，后面接dropout以及relu激活
            fc = tf.layers.dense(last, self.config.hidden_dim, name='fc1')
            # if self.input_istrain>0:
            #     fc = tf.contrib.layers.dropout(fc,
            #         self.config.dropout_keep_prob)
            # else:
            #     fc = tf.contrib.layers.dropout(fc,1.0)
            tf.cond(self.input_istrain>0,lambda :tf.contrib.layers.dropout(fc,
                    self.config.dropout_keep_prob),lambda :tf.contrib.layers.dropout(fc,1.0))
            fc = tf.nn.relu(fc)

            # 分类器
            self.logits = tf.layers.dense(fc, self.config.num_classes,
                name='fc2')
            self.pred_y = tf.nn.softmax(self.logits)

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
            correct_pred = tf.equal(tf.argmax(self.input_y, 1),
                tf.argmax(self.pred_y, 1))
            self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
