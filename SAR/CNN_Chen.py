import tensorflow as tf
import math
import scipy.misc
import numpy as np
DROPOUT = 0.7  # dropout比例
NUM_CLASSES = 10  # 类别数一共是10类
### define gloabal parameter
conv_1_trans_1 = 0
conv_1_trans_2 = 0
conv_2_trans_1 = 0
conv_2_trans_2 = 0
conv_3_trans_1 = 0
conv_3_trans_2 = 0
conv_4_trans_1 = 0
conv_4_trans_2 = 0
conv_5_trans_1 = 0
conv_5_trans_2 = 0

def variable_summaries(name, var):
    '''
    对变量var进行汇总(summaries),以便实现可视化
    :param name: 可视化的名称
    :param var: 进行可视化的变量
    :return: 无返回值
    '''
    tf.summary.scalar(name, var)
    tf.summary.histogram(name, var)


def Inference(img_batch, batch_size, regularizer, train):
    '''
    搭建神经网络的网络结构
    :param img_batch: 一个batch_size的输入图像
    :param regularizer: 正则化标志,不为None即对权重每一层卷积核的weight执行正则化
    :return: logits: 神经网络前向传播结果
    '''
    with tf.variable_scope('conv1') as scope:

        weights = tf.get_variable(name='weights', shape=[5, 5, 1, 16], dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.01))
        tf.summary.histogram('conv1'+'weights', weights)
        conv = tf.nn.conv2d(img_batch, weights, [1,1,1,1], padding='VALID')  # padding='VALID',卷积层缩小feature map的大小
        biases = tf.Variable(tf.constant(0.1, shape=[16]), name='biases')
        conv1 = tf.nn.relu(tf.nn.bias_add(conv, biases), name=scope.name)
        #variable_summaries('conv1'+'activation', conv1)
        # 如果正则化项为真,将权重weights的L2范数加入到losses集合中.
        if regularizer is not None:
            tf.add_to_collection('losses', regularizer(weights))
        if train is not True:
            global conv_1_trans_1
            global conv_1_trans_2
            conv_1_trans = tf.transpose(conv1, [3,1,2,0])
            conv_1_trans_1 = tf.reshape(conv_1_trans[0],[84,84])
            conv_1_trans_2 = tf.reshape(conv_1_trans[1],[84,84])
            tf.summary.image('image1', conv_1_trans, 16)
            #print(conv_1_trans[0].shape)
            #print(gray_i.shape)

            #a = gray_i.eval()
            #print(np.array(gray_i).shape)
            #print(gray_i)
            #scipy.misc.imsave('outfile.jpg',gray_i)
        # max pooling1
        pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1,], strides=[1, 2, 2, 1], padding='SAME', name = 'pool1')

    with tf.variable_scope('conv2') as scope:

        weights = tf.get_variable(name='weights', shape=[5, 5, 16, 32], dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.01))
        conv = tf.nn.conv2d(pool1, weights, [1, 1, 1, 1], padding='VALID')
        biases = tf.Variable(tf.constant(0.1, shape=[32]), name = 'biases')
        conv2 = tf.nn.relu(tf.nn.bias_add(conv, biases), name = scope.name)
        #variable_summaries('conv2'+'activation', conv2)
        # 如果正则化项为真,将权重weights的L2范数加入到losses集合中.
        if regularizer is not None:
            tf.add_to_collection('losses', regularizer(weights))
        if train is not True:
            global conv_2_trans_1
            global conv_2_trans_2
            conv_2_trans = tf.transpose(conv2, [3,1,2,0])
            print(conv_2_trans[0].shape)
            conv_2_trans_1 = tf.reshape(conv_2_trans[0],[38,38])
            #conv_2_trans_1 = tf.image.resize_images(conv_2_trans_1, [84,84],method=0)
            conv_2_trans_2 = tf.reshape(conv_2_trans[1],[38,38])
            #conv_2_trans_2 = tf.image.resize_images(conv_2_trans_2, [84,84],method=0)
            tf.summary.image('image2', conv_2_trans, 32)

        # max pooling2
        pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name = 'pool2')

    with tf.variable_scope('conv3') as scope:

        weights = tf.get_variable(name='weights', shape=[6, 6, 32, 64], dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.01))
        conv = tf.nn.conv2d(pool2, weights, [1, 1, 1, 1], padding='VALID')
        biases = tf.Variable(tf.constant(0.1, shape=[64]), name = 'biases')
        conv3 = tf.nn.relu(tf.nn.bias_add(conv, biases), name = scope.name)
        #variable_summaries('conv3'+'activation', conv3)
        # 如果正则化项为真,将权重weights的L2范数加入到losses集合中.
        if regularizer is not None:
            tf.add_to_collection('losses', regularizer(weights))
        if train is not True:
            conv_3_trans = tf.transpose(conv3, [3, 1, 2, 0])
            print(conv_3_trans[0].shape)
            global conv_3_trans_1
            global conv_3_trans_2
            print(conv_3_trans[0].shape)
            conv_3_trans_1 = tf.reshape(conv_3_trans[0],[14,14])
            conv_3_trans_2 = tf.reshape(conv_3_trans[1],[14,14])
            tf.summary.image('image3', conv_3_trans, 64)

        #max pooling3
        pool3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name = 'pool3')

    with tf.variable_scope('conv4') as scope:

        weights = tf.get_variable(name='weights', shape=[5, 5, 64, 128], dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.01))
        conv = tf.nn.conv2d(pool3, weights, [1, 1, 1, 1], padding='VALID')
        biases = tf.Variable(tf.constant(0.1, shape=[128]), name = 'biases')
        conv4 = tf.nn.relu(tf.nn.bias_add(conv, biases), name = scope.name)
        #variable_summaries('conv4'+'activation', conv4)
        # 如果正则化项为真,将权重weights的L2范数加入到losses集合中.
        if regularizer is not None:
            tf.add_to_collection('losses', regularizer(weights))

        # no max_pooling but dropout, dropout is performed only in train
        if train == True:
            conv4 = tf.nn.dropout(conv4, DROPOUT)
        else:
            conv_4_trans = tf.transpose(conv4, [3, 1, 2, 0])
            global conv_4_trans_1
            global conv_4_trans_2
            print(conv_4_trans[0].shape)
            conv_4_trans_1 = tf.reshape(conv_4_trans[0],[3,3])
            conv_4_trans_2 = tf.reshape(conv_4_trans[1],[3,3])
            print(conv_4_trans[0].shape)
            tf.summary.image('image4', conv_4_trans, 128)
        #variable_summaries('conv4'+'dropout', conv4)

    with tf.variable_scope('conv5') as scope:

        weights = tf.get_variable(name='weights', shape=[3, 3, 128, NUM_CLASSES], dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.01))
        conv = tf.nn.conv2d(conv4, weights, [1, 1, 1, 1], padding='VALID')
        biases = tf.Variable(tf.constant(0.1, shape=[NUM_CLASSES]), name = 'biases')
        logits = tf.nn.bias_add(conv, biases)
        # 如果正则化项为真,将权重weights的L2范数加入到losses集合中.
        if regularizer is not None:
            tf.add_to_collection('losses', regularizer(weights))
        if train is not True:
            conv_5_trans = tf.transpose(logits, [3, 1, 2, 0])
            global conv_5_trans_1
            global conv_5_trans_2
            print(conv_5_trans[0].shape)
            conv_5_trans_1 = tf.reshape(conv_5_trans[0],[1,1])
            conv_5_trans_2 = tf.reshape(conv_5_trans[1],[1,1])
            print(conv_5_trans[0].shape)
            tf.summary.image('image5', conv_5_trans, 10)

    logits = tf.reshape(logits, shape=[batch_size, NUM_CLASSES])

    return logits


def Loss(logits, labels):
    '''
    计算交叉熵损失cross entropy与权重衰减损失的和,即总损失
    :param logits: Inference()函数生成的网络预测值,它的shape是[batch_size, num_classes]
    :param labels: 真实的标签,它的shape是[batch_size]
    :return: 总损失：如果进行正则化处理的话：cross entropy + regularizer(weights)
                    如果不进行正则化处理,总损失就是交叉熵损失
    '''
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels = labels, logits = logits, name = 'cross_entropy_per_example'
    )

    cross_entropy = tf.reduce_mean(cross_entropy, name = 'cross_entropy')
    # 将所有loss放入集合’losses‘中
    tf.add_to_collection('losses', cross_entropy)

    total_loss = tf.add_n(tf.get_collection('losses'), name = 'total_loss')

    return total_loss
