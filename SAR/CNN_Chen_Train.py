# -*- coding:UTF-8 -*-
from PIL import Image
from datetime import datetime
import time
import os
import sys
import numpy as np
import tensorflow as tf

import CNN_Chen  # 导入神经网络结构的文件
import Generate_Batch  # 导入读取数据并生成batch文件

IMAGE_SIZE = 88  # 输入图像大小为88*88
BATCH_SIZE = 128  # batch的大小是128

MAX_STEPS = 70001  # 总计训练一百万次/////7W;7W零1以保存最后一次的训练模型
NUM_EPOCHES_PER_DECAY = 50  # 每50个epoches学习率调整一次；Epochs after which learning rate

INITIAL_LEARNING_RATE = 0.01  # 初始学习率；Initial learning rate   第一次是10e-4
LEARNING_RATE_DECAY_FACTOR = 0.1  # 学习率衰减因子；Learning rate decay factor
REGULARIZATION_RATE = 0.0001  # 总损失中权重衰减的正则化系数,可以减小网络的过拟合;L2正则化系数，仅对权重进行正则化约束，减少网络过拟合程度.

# 总计训练数据量是41205=2747*3*5
# 3代表128*128大小的原图、上下翻转以及水平翻转；5代表中间88*88，以及上下左右四个顶点88*88的图
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 41205
# 原始训练集数据量是2747，对其进行上下翻转和水平镜像，
# 总共得到8241幅大小为128*128的灰度图像；随后从128*128图像
# 才裁剪大小为88*88图像作为神经网络的输入，每幅128图像可以生成
# 1681幅88图像。原文中，每一类图像做如下类似的操作：
# 如：BMP-2有233幅图像，从233*1681中随机选择2700幅图像进行训练。

readpath = '/home/wangyang/下载/SAR/test/Label'
filename = readpath + '/train.tfrecords'
save_path = '/home/wangyang/下载/SAR/model'
save_name = 'model.ckpt'


def Train(total_loss, global_step):
    '''
    创建一个优化器，并应用于所有可训练的变量；为所有的可训练变量添加滑动平均操作(还未实现滑动平均)
    :param total_loss: Total loss from loss()
    :param global_step: Interger Variable counting the numbers of training steps processed
    :return:train_op： op for training
    '''

    num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / BATCH_SIZE  # 训练数据跑一个epoch需要的训练次数
    decay_steps = int(num_batches_per_epoch * NUM_EPOCHES_PER_DECAY)  # 每训练50个epoch,学习率下降一次
    # 学习率呈阶梯状衰减
    lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                    global_step,
                                    decay_steps,
                                    LEARNING_RATE_DECAY_FACTOR,
                                    staircase=True)  # 学习率呈阶梯状衰减,每50个周期衰减一次,每次下降0.1

    tf.summary.scalar('learning_rate', lr)  # 记录学习率的变化

    #  train_step = tf.train.GradientDescentOptimizer(lr).minimize(total_loss, global_step = global_step)  # 优化目标函数,即最小化损失
    train_step = tf.train.MomentumOptimizer(lr, 0.9).minimize(total_loss, global_step = global_step)  # 采用Momentum的方法进行优化

    return train_step


global_step = tf.Variable(0, trainable=False)

image, label = Generate_Batch.read_data_sets(filename=filename, image_size=IMAGE_SIZE)

# image = tf.image.per_image_standardization(image)  # 均值为0,标准差为1-------训练不收敛

image = image / 255.  # 唯一的图像预处理操作,像素灰度值归一化到[0,1]之间

# image_mean = tf.reduce_mean(image)

# image = image - image_mean  # 去均值操作,一幅图像每一个像素点减去自身图像像素的均值,实现0均值.

image_batch, label_batch = Generate_Batch.generate_batch(image=image, label=label, batch_size=BATCH_SIZE)

regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
# 训练的时候选择加入权重的L2正则项,测试的时候不加
logits = CNN_Chen.Inference(image_batch, BATCH_SIZE, regularizer, train=True)

total_loss = CNN_Chen.Loss(logits=logits, labels=label_batch)

train_step = Train(total_loss, global_step)

saver = tf.train.Saver()

merged = tf.summary.merge_all()

with tf.Session() as sess:

    init = tf.global_variables_initializer()
    sess.run(init)

    summary_writer = tf.summary.FileWriter(logdir='/home/f523/LJY/SAR/MSTAR/event', graph=sess.graph)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    print(global_step.eval())
    print('---------------------------------------------------------')
    print('---------------------------------------------------------')
    print('---------------------------------------------------------')
    print('---------------------------------------------------------')
    print('---------------------------------------------------------')

    start_time = time.time()

    for step in range(MAX_STEPS):

        _, loss_value = sess.run([train_step, total_loss])

        if step % 100 == 0:
            current_time = time.time()
            duration = current_time - start_time
            start_time = current_time

            format_str = ('%s: step %d, loss = %.4f')
            print(format_str % (datetime.now(), step, loss_value))
            # 运行所有日志生成操作，得到这次运行的日志，并将所有日志写入文件，TensorBoard程序就可以得到这次运行所对应的运行信息.
            if step % 1000 == 0:
                summary = sess.run(merged)
                summary_writer.add_summary(summary, step)
                if step % 10000 == 0:
                    saver.save(sess, save_path=os.path.join(save_path, save_name))
                    print('------Save Model %d time(s)------' % (step / 10000))

    print('Training Models Finished!!!!!')

    coord.request_stop()
    coord.join(threads)

summary_writer.close()
