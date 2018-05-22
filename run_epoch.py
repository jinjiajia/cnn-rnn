#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import print_function

from rnn_model import *
from cnn_model import *
from configuration import *
#from data.cnews_loader import *#original
from data.xlsx_loader import *#enjoyer test


import time
from datetime import timedelta
save_dir = 'checkpoints/textcnn'
save_path = os.path.join(save_dir, 'best_validation')  # 最佳验证结果保存路径

def run_epoch(cnn=True):
    # 载入数据
    print('Loading data...')
    start_time = time.time()
    # trainlines,trainlab,testlines,testlab=wenti_read_file('wenti1.xlsx')
    trainlines,testlines,trainlab,testlab=wenti_read_file('wenti1.xlsx')
    if not os.path.exists('data/cnews/vocab_cnews.txt'):
        #build_vocab('data/cnews/cnews.train.txt')
        build_vocab(trainlines)
    # x_train, y_train, x_test, y_test, words = preocess_file(trainlines,trainlab,testlines,testlab)
    x_train, y_train, x_test, y_test, words = preocess_file(trainlines, testlines, trainlab, testlab)
    # print(y_test.shape[1])
    # exit()
    num_class=y_test.shape[1]
    if cnn:
        print('Using CNN model...')
        config = TCNNConfig()
        config.vocab_size = len(words)
        config.num_classes=num_class
        model = TextCNN(config)
        tensorboard_dir = 'tensorboard/textcnn'
    else:
        print('Using RNN model...')
        config = TRNNConfig()
        config.vocab_size = len(words)
        config.num_classes=num_class
        model = TextRNN(config)
        tensorboard_dir = 'tensorboard/textrnn'

    end_time = time.time()
    time_dif = end_time - start_time
    time_dif = timedelta(seconds=int(round(time_dif)))
    print('Time usage:', time_dif)
    # global_step=tf.Variable(0,trainable=False)
    # add_step=global_step.assign_add(1)
    # update_rate = tf.train.exponential_decay(config.learning_rate,
    #                                        global_step=global_step,
    #                                        decay_steps=100,decay_rate=0.9)

    print('Constructing TensorFlow Graph...')
    session = tf.Session()
    session.run(tf.global_variables_initializer())

    # 配置 tensorboard
    tf.summary.scalar("loss", model.loss)
    tf.summary.scalar("accuracy", model.acc)

    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)

    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter(tensorboard_dir)
    writer.add_graph(session.graph)

    # 配置 Saver
    saver = tf.train.Saver()
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 生成批次数据
    print('Generating batch...')
    batch_train = batch_iter(list(zip(x_train, y_train)),
        config.batch_size, config.num_epochs)

    def feed_data(batch,step,istraining):
        """准备需要喂入模型的数据"""
        x_batch, y_batch = zip(*batch)
        feed_dict = {
            model.input_x: x_batch,
            model.input_y: y_batch,
            model.input_step: step,
            model.input_istrain: istraining

        }
        return feed_dict, len(x_batch)

    def evaluate(x_, y_):
        """
        模型评估
        一次运行所有的数据会OOM，所以需要分批和汇总
        """
        batch_eval = batch_iter(list(zip(x_, y_)), 128, 1)

        total_loss = 0.0
        total_acc = 0.0
        cnt = 0
        for batch in batch_eval:
            feed_dict, cur_batch_len = feed_data(batch,0,istraining=-1)
            loss, acc = session.run([model.loss, model.acc],
                feed_dict=feed_dict)
            total_loss += loss * cur_batch_len
            total_acc += acc * cur_batch_len
            cnt += cur_batch_len

        return total_loss / cnt, total_acc / cnt

    # 训练与验证
    print('Training and evaluating...')
    total_batch = 0  # 总批次
    best_acc_val = 0.0  # 最佳验证集准确率
    last_improved = 0  # 记录上一次提升批次
    start_time = time.time()
    print_per_batch = config.print_per_batch
    for i, batch in enumerate(batch_train):
        feed_dict, _ = feed_data(batch,i,istraining=config.istraining)
        # session.run(add_step,update_rate)


        if i % 5 == 0:  # 每5次将训练结果写入tensorboard scalar
            s = session.run(merged_summary, feed_dict=feed_dict)
            writer.add_summary(s, i)

        if i % print_per_batch == print_per_batch - 1:  # 每200次输出在训练集和验证集上的性能
            loss_train, acc_train = session.run([model.loss, model.acc],
                feed_dict=feed_dict)
            loss_val, acc_val = evaluate(x_test, y_test)
            if acc_val > best_acc_val:
                # 保存最好结果
                best_acc_val = acc_val
                last_improved = total_batch
                saver.save(sess=session, save_path=save_path)
                improved_str = '*'
            else:
                improved_str = ''

            # 时间
            end_time = time.time()
            time_dif = end_time - start_time
            time_dif = timedelta(seconds=int(round(time_dif)))

            # msg = 'Iter: d%, Train Loss: f%, Train Acc: f%, Time: d%'
            print("Iter: %d, Train Loss: %f, Train Acc: %f"%((i + 1), loss_train,acc_train))

        session.run(model.optim, feed_dict=feed_dict)  # 运行优化

    # 最后在测试集上进行评估
    print('Evaluating on test set...')
    loss_test, acc_test = evaluate(x_test, y_test)
    msg = 'Test Loss: {0:>6.2}, Test Acc: {1:>7.2%}'
    print(msg.format(loss_test, acc_test))

    session.close()

if __name__ == '__main__':
    run_epoch(cnn=True)
