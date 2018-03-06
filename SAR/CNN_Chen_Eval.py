# -*- coding:UTF-8 -*-
import tensorflow as tf
import numpy as np
import os
import math
from datetime import datetime
import scipy.misc
import CNN_Chen
import Generate_Batch
from easydict import EasyDict as edict
from PIL import Image, ImageTk
from scipy import misc
import sys
#path = '/home/wangyang/桌面/interface'
#sys.path.append(path)
#import interface

Test_num_examples = [274, 195, 274, 195, 196, 274, 273, 196, 274, 274]  # 0类(0_2S1)的测试样本个数
Test_Batch_Size = 1  # 测试图像时要保证对每一幅图像进行测试,因此测试的batch应为1
IMAGE_SIZE = 88
index_predict = ''
final_results = np.zeros((10,10))

filename = '/home/wangyang/下载/SAR/test/Label/test_2.tfrecords'

#path_single = "/home/wangyang/桌面/interface/SAR/test/Image_88/0/Center_0_HB14931.JPG"
#path_model = '/home/wangyang/下载/SAR/model/model.ckpt'
def evaluation_single_image(logit, saver, merged, path_model):
    with tf.Session() as sess:
        #ckpt = tf.train.get_checkpoint_state('/home/wangyang/下载/SAR/model/model.ckpt')
        #if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, path_model)
            #print(ckpt.model_checkpoint_path)
        #else:
        #    print('No checkpoint file found!')
        #    return
        summary_writer = tf.summary.FileWriter('/home/wangyang/下载/SAR/event_single', sess.graph)
        true_count = 0
        #print(interface.aaa)

        conv_1_trans_1 = CNN_Chen.conv_1_trans_1.eval()
        conv_1_trans_2 = CNN_Chen.conv_1_trans_2.eval()
        conv_2_trans_1 = CNN_Chen.conv_2_trans_1.eval()
        conv_2_trans_2 = CNN_Chen.conv_2_trans_2.eval()
        conv_3_trans_1 = CNN_Chen.conv_3_trans_1.eval()
        conv_3_trans_2 = CNN_Chen.conv_3_trans_2.eval()
        conv_4_trans_1 = CNN_Chen.conv_4_trans_1.eval()
        conv_4_trans_2 = CNN_Chen.conv_4_trans_2.eval()
        conv_5_trans_1 = CNN_Chen.conv_5_trans_1.eval()
        conv_5_trans_2 = CNN_Chen.conv_5_trans_2.eval()
        #print(np.array(conv_1_trans_1).shape)
        scipy.misc.imsave('feature_map/layer1_1.jpg',conv_1_trans_1)
        scipy.misc.imsave('feature_map/layer1_2.jpg',conv_1_trans_2)
        scipy.misc.imsave('feature_map/layer2_1.jpg',conv_2_trans_1)
        scipy.misc.imsave('feature_map/layer2_2.jpg',conv_2_trans_2)
        scipy.misc.imsave('feature_map/layer3_1.jpg',conv_3_trans_1)
        scipy.misc.imsave('feature_map/layer3_2.jpg',conv_3_trans_2)
        scipy.misc.imsave('feature_map/layer4_1.jpg',conv_4_trans_1)
        scipy.misc.imsave('feature_map/layer4_2.jpg',conv_4_trans_2)
        scipy.misc.imsave('feature_map/layer5_1.jpg',conv_5_trans_1)
        scipy.misc.imsave('feature_map/layer5_2.jpg',conv_5_trans_2)

        layer2_1 = misc.imread('/home/wangyang/桌面/interface/feature_map/layer2_1.jpg')
        layer2_1_84 = misc.imresize(layer2_1,[84, 84],interp = "nearest")
        misc.imsave('/home/wangyang/桌面/interface/feature_map/layer2_1_84.jpg',np.array(layer2_1_84))

        layer2_2 = misc.imread('/home/wangyang/桌面/interface/feature_map/layer2_2.jpg')
        layer2_2_84 = misc.imresize(layer2_2,[84, 84],interp = "nearest")
        misc.imsave('/home/wangyang/桌面/interface/feature_map/layer2_2_84.jpg',np.array(layer2_2_84))

        layer3_1 = misc.imread('/home/wangyang/桌面/interface/feature_map/layer3_1.jpg')
        layer3_1_84 = misc.imresize(layer3_1,[84, 84],interp = "nearest")
        misc.imsave('/home/wangyang/桌面/interface/feature_map/layer3_1_84.jpg',np.array(layer3_1_84))

        layer3_2 = misc.imread('/home/wangyang/桌面/interface/feature_map/layer3_2.jpg')
        layer3_2_84 = misc.imresize(layer3_2,[84, 84],interp = "nearest")
        misc.imsave('/home/wangyang/桌面/interface/feature_map/layer3_2_84.jpg',np.array(layer3_2_84))

        layer4_1 = misc.imread('/home/wangyang/桌面/interface/feature_map/layer4_1.jpg')
        layer4_1_84 = misc.imresize(layer4_1,[84, 84],interp = "nearest")
        misc.imsave('/home/wangyang/桌面/interface/feature_map/layer4_1_84.jpg',np.array(layer4_1_84))

        layer4_2 = misc.imread('/home/wangyang/桌面/interface/feature_map/layer4_2.jpg')
        layer4_2_84 = misc.imresize(layer4_2,[84, 84],interp = "nearest")
        misc.imsave('/home/wangyang/桌面/interface/feature_map/layer4_2_84.jpg',np.array(layer4_2_84))

        layer5_1 = misc.imread('/home/wangyang/桌面/interface/feature_map/layer5_1.jpg')
        layer5_1_84 = misc.imresize(layer5_1,[84, 84],interp = "nearest")
        misc.imsave('/home/wangyang/桌面/interface/feature_map/layer5_1_84.jpg',np.array(layer5_1_84))

        layer5_2 = misc.imread('/home/wangyang/桌面/interface/feature_map/layer5_2.jpg')
        layer5_2_84 = misc.imresize(layer5_2,[84, 84],interp = "nearest")
        misc.imsave('/home/wangyang/桌面/interface/feature_map/layer5_2_84.jpg',np.array(layer5_2_84))

        #a = Image.open('feature_map/layer1_1.jpg')
        #step = 1
        # 用于记录误分类测试样本其logit最大值的索引位置
        index = tf.argmax(logit, 1)
        global index_predict
        index_pos, summary = sess.run([index, merged])
        index_predict = index_pos[0]
        print(index_pos[0])
        # 将top_k_op操作和tf.argmax(logit,1)同时执行才能得到正确的准确率,
        # 否则若先执行top_k_op,而后在下面的if语句中再执行index.eval()【这里index是Tensor】,
        # Tensor.eval()会将这个生成这个张量的所有变量执行一次,即logit又执行一次,执行的是下一个图像，所以准确率会有偏差！！！！！这里的一个坑
        #print(step, predictions[0])
        #if predictions[0] != True:
        #print('第1类样本被误分类为第%d类样本' % (index_pos[0]))
        #true_count += np.sum(predictions)
        summary_writer.add_summary(summary)
        #step += 1

def evaluation_once(logit, saver, top_k_op, merged,path_model):

    with tf.Session() as sess:
        #ckpt = tf.train.get_checkpoint_state('/home/wangyang/下载/SAR/model/model.ckpt')
        #if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, path_model)
            #print(ckpt.model_checkpoint_path)
        #else:
        #    print('No checkpoint file found!')
        #    return

        summary_writer = tf.summary.FileWriter('/home/wangyang/下载/SAR/event', sess.graph)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        true_count = 0
        step = 1
        for i in range(10):
            for j in range(10):
                final_results[i][j] = 0
        # 用于记录误分类测试样本其logit最大值的索引位置
        index = tf.argmax(logit, 1)
        """
        while step <= Test_num_examples[0] and not coord.should_stop():
            predictions, index_pos, summary = sess.run([top_k_op, index, merged])
            # 将top_k_op操作和tf.argmax(logit,1)同时执行才能得到正确的准确率,
            # 否则若先执行top_k_op,而后在下面的if语句中再执行index.eval()【这里index是Tensor】,
            # Tensor.eval()会将这个生成这个张量的所有变量执行一次,即logit又执行一次,执行的是下一个图像，所以准确率会有偏差！！！！！这里的一个坑
            print(step, predictions[0])
            if predictions[0] != True:
                print('第1类样本中的第%d个样本被误分类为第%d类样本' % (step, index_pos[0]))
                final_results[0][index_pos[0]] = final_results[0][index_pos[0]]+1
            if predictions[0] == True:
                final_results[0][0] = final_results[0][0]+1
            true_count += np.sum(predictions)
            summary_writer.add_summary(summary, step)
            step += 1
        """
        for i in range(10):
            step = 1
            while step <= Test_num_examples[i] and not coord.should_stop():
                predictions, index_pos, summary = sess.run([top_k_op, index, merged])
                # 将top_k_op操作和tf.argmax(logit,1)同时执行才能得到正确的准确率,
                # 否则若先执行top_k_op,而后在下面的if语句中再执行index.eval()【这里index是Tensor】,
                # Tensor.eval()会将这个生成这个张量的所有变量执行一次,即logit又执行一次,执行的是下一个图像，所以准确率会有偏差！！！！！这里的一个坑
                print(step, predictions[0])
                if predictions[0] != True:
                    print('第1类样本中的第%d个样本被误分类为第%d类样本' % (step, index_pos[0]))
                    final_results[i][index_pos[0]] = final_results[i][index_pos[0]]+1
                if predictions[0] == True:
                    final_results[i][i] = final_results[i][i]+1
                true_count += np.sum(predictions)
                summary_writer.add_summary(summary, step)
                step += 1

        print(final_results)
        precision = true_count / Test_num_examples[1]
        print('%s: precision = %.4f' % (datetime.now(), precision))

        #print(step-1)
        print(true_count)

        coord.request_stop()
        coord.join(threads)


def evaluation(path_model,path_record):
    with tf.Graph().as_default():

        image, label = Generate_Batch.read_data_sets(filename=path_record, image_size=IMAGE_SIZE)

        image = image / 255.

        image_batch, label_batch = Generate_Batch.generate_batch(image, label, Test_Batch_Size)
        print(label_batch)
        logits = CNN_Chen.Inference(image_batch, Test_Batch_Size, None, train=False)

        merged_test = tf.summary.merge_all()
        # logits = tf.nn.softmax(logits)  # 无需进行softmax,因为比较最大值的位置即可
        top_k_op = tf.nn.in_top_k(logits, label_batch, 1)

        saver = tf.train.Saver()

        evaluation_once(logits, saver, top_k_op, merged_test,path_model)

def evaluation_single(path_model,path_image):
    with tf.Graph().as_default():
        print(path_image)
        image, label = Generate_Batch.read_single_image(filename=path_image, image_size=IMAGE_SIZE)

        image = image / 255.
        image = tf.reshape(image, [1, 88, 88, 1])
        #image_batch, label_batch = Generate_Batch.generate_batch(image, label, Test_Batch_Size)

        logits = CNN_Chen.Inference(image, Test_Batch_Size, None, train=False)

        merged_test = tf.summary.merge_all()

        # logits = tf.nn.softmax(logits)  # 无需进行softmax,因为比较最大值的位置即可

        #top_k_op = tf.nn.in_top_k(logits, label_batch, 1)
        #print(CNN_Chen.conv1/transpose)
        saver = tf.train.Saver()
        evaluation_single_image(logits, saver, merged_test, path_model)
#def main(argv=None):
#    pass
    #evaluation()
    #evaluation_single()


#if __name__ == '__main__':
#    tf.app.run()
