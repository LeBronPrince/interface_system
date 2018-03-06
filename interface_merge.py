# -*- coding:UTF-8 -*-
import tensorflow as tf
import numpy as np
import os
import math
from datetime import datetime
import scipy.misc
from easydict import EasyDict as edict
from PIL import Image, ImageTk
from scipy import misc
import sys
path = os.getcwd()+'/SAR'
sys.path.append(path)
import Generate_Batch
import CNN_Chen
#import interface
from tkinter import *
from PIL import Image, ImageTk
from tkinter.filedialog import askdirectory,askopenfilename
from tkinter import messagebox
from tkinter.scrolledtext import ScrolledText


Test_num_examples = [274, 195, 274, 195, 196, 274, 273, 196, 274, 274]  # 0类(0_2S1)的测试样本个数
Test_Batch_Size = 1  # 测试图像时要保证对每一幅图像进行测试,因此测试的batch应为1
IMAGE_SIZE = 88
index_predict = ''
final_results = np.zeros((10,10))
path_event_single = os.getcwd()+'/SAR/event_single'
path_event_single = os.getcwd()+'/SAR/event'


array_class = ['2S1','BMP2','BRDM2','BTR60','BTR70','D7','T62','T72','ZIL131','ZSU234']
#filename = '/home/wangyang/下载/SAR/test/Label/test_2.tfrecords'

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
        summary_writer = tf.summary.FileWriter(path_event_single, sess.graph)
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

        layer2_1 = misc.imread('feature_map/layer2_1.jpg')
        layer2_1_84 = misc.imresize(layer2_1,[84, 84],interp = "nearest")
        misc.imsave('feature_map/layer2_1_84.jpg',np.array(layer2_1_84))

        layer2_2 = misc.imread('feature_map/layer2_2.jpg')
        layer2_2_84 = misc.imresize(layer2_2,[84, 84],interp = "nearest")
        misc.imsave('feature_map/layer2_2_84.jpg',np.array(layer2_2_84))

        layer3_1 = misc.imread('feature_map/layer3_1.jpg')
        layer3_1_84 = misc.imresize(layer3_1,[84, 84],interp = "nearest")
        misc.imsave('feature_map/layer3_1_84.jpg',np.array(layer3_1_84))

        layer3_2 = misc.imread('feature_map/layer3_2.jpg')
        layer3_2_84 = misc.imresize(layer3_2,[84, 84],interp = "nearest")
        misc.imsave('feature_map/layer3_2_84.jpg',np.array(layer3_2_84))

        layer4_1 = misc.imread('feature_map/layer4_1.jpg')
        layer4_1_84 = misc.imresize(layer4_1,[84, 84],interp = "nearest")
        misc.imsave('feature_map/layer4_1_84.jpg',np.array(layer4_1_84))

        layer4_2 = misc.imread('feature_map/layer4_2.jpg')
        layer4_2_84 = misc.imresize(layer4_2,[84, 84],interp = "nearest")
        misc.imsave('feature_map/layer4_2_84.jpg',np.array(layer4_2_84))

        layer5_1 = misc.imread('feature_map/layer5_1.jpg')
        layer5_1_84 = misc.imresize(layer5_1,[84, 84],interp = "nearest")
        misc.imsave('feature_map/layer5_1_84.jpg',np.array(layer5_1_84))

        layer5_2 = misc.imread('feature_map/layer5_2.jpg')
        layer5_2_84 = misc.imresize(layer5_2,[84, 84],interp = "nearest")
        misc.imsave('feature_map/layer5_2_84.jpg',np.array(layer5_2_84))

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

        summary_writer = tf.summary.FileWriter(path_event_single, sess.graph)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        true_count = 0
        step = 1
        for i in range(10):
            for j in range(10):
                final_results[i][j] = 0
        # 用于记录误分类测试样本其logit最大值的索引位置
        index = tf.argmax(logit, 1)
        #scrotext = ScrolledText(frame4,width=60,height=20,font=("Arial",11))
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
            scrotext.insert(END,'第%d类检测出错的为第  '%(i))
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
                    scrotext.insert(END,step)
                    scrotext.insert(END,' ')
                if predictions[0] == True:
                    final_results[i][i] = final_results[i][i]+1
                true_count += np.sum(predictions)
                summary_writer.add_summary(summary, step)
                step += 1
            scrotext.insert(END,'共%d张\n'%(Test_num_examples[i]-final_results[i][i]))


        scrotext.insert(END,'检测10类图片%d张图片,'%(np.sum(Test_num_examples)))
        scrotext.insert(END,'正确率为%.4f\n'%(true_count/(np.sum(Test_num_examples))))
        scrotext.insert(END,'\n')
        #scrotext.place(x=300,y=360)
        print(final_results)
        precision = true_count / Test_num_examples[1]
        print('%s: precision = %.4f' % (datetime.now(), precision))

        #print(step-1)
        #print(true_count)

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
        image = tf.reshape(image, [1, IMAGE_SIZE, IMAGE_SIZE, 1])
        #image_batch, label_batch = Generate_Batch.generate_batch(image, label, Test_Batch_Size)

        logits = CNN_Chen.Inference(image, Test_Batch_Size, None, train=False)

        merged_test = tf.summary.merge_all()

        # logits = tf.nn.softmax(logits)  # 无需进行softmax,因为比较最大值的位置即可

        #top_k_op = tf.nn.in_top_k(logits, label_batch, 1)
        #print(CNN_Chen.conv1/transpose)
        saver = tf.train.Saver()
        evaluation_single_image(logits, saver, merged_test, path_model)




model_config_1 = edict()
model_config_1.image_model = 'SAR/model_information/model1/model.jpg'
model_config_1.image_loss ='SAR/model_information/model1/lr.png'
model_config_1.image_lr = 'SAR/model_information/model1/loss.png'
model_config_1.model_checkpoint_path = 'SAR/model/model.ckpt'

model_config_2 = edict()
model_config_2.image_model = 'SAR/model_information/model2/model.jpg'
model_config_2.image_loss ='SAR/model_information/model2/lr.png'
model_config_2.image_lr = 'SAR/model_information/model2/loss.png'
model_config_2.model_checkpoint_path = 'SAR/model2/model.ckpt'

model_config_3 = edict()
model_config_3.image_model = 'SAR/model_information/image_model1.jpg'
model_config_3.image_loss =''
model_config_3.image_lr = ''
model_config_3.model_checkpoint_path = 'SAR/model2/model.ckpt'

model_select = model_config_1
tk_model_select= 1
tk_loss_select= 1
tk_lr_select= 1
path_single = "SAR/test/Image_88/5/Center_5_HB14932.JPG"
path_record = "SAR/test/Label/all.tfrecords"
#path_model = '/home/wangyang/下载/SAR/model/model.ckpt'
#model_name="model"
root=Tk()
root.geometry("1040x600")
root.title("test system")
root.resizable(width=False,height=False)
frame1=Frame(root,width=1040,height=600,bg='LightSkyBlue')
frame2=Frame(root,width=1040,height=600,bg='LightSkyBlue')
frame3=Frame(root,width=1040,height=600,bg='LightSkyBlue')
frame4=Frame(root,width=1040,height=600,bg='LightSkyBlue')
frame1.pack()

def to_frame1():
	frame3.pack_forget()
	frame2.pack_forget()
	frame4.pack_forget()
	frame1.pack()
def to_frame2():
	frame3.pack_forget()
	frame1.pack_forget()
	frame4.pack_forget()
	frame2.pack()
def to_frame3():
	frame1.pack_forget()
	frame2.pack_forget()
	frame4.pack_forget()
	frame3.pack()
def to_frame4():
	frame1.pack_forget()
	frame2.pack_forget()
	frame3.pack_forget()
	frame4.pack()

def select_mode(m):
	global model_select
	global tk_model_select
	global tk_lr_select
	global tk_loss_select
	if m==1:
		model_select = model_config_1
		image_model_select = Image.open(model_select.image_model)
		print(model_select.image_model)
		tk_model_select = ImageTk.PhotoImage(image_model_select)
		image_lr_select = Image.open(model_config_1.image_lr)
		tk_lr_select = ImageTk.PhotoImage(image_lr_select)

		image_loss_select = Image.open(model_config_1.image_loss)
		tk_loss_select = ImageTk.PhotoImage(image_loss_select)
		model_num.set('model1')

	if m==2:
		model_select = model_config_2
		image_model_select = Image.open(model_select.image_model)
		print(model_select.image_model)
		tk_model_select = ImageTk.PhotoImage(image_model_select)

		image_lr_select = Image.open(model_config_1.image_lr)
		tk_lr_select = ImageTk.PhotoImage(image_lr_select)

		image_loss_select = Image.open(model_config_1.image_loss)
		tk_loss_select = ImageTk.PhotoImage(image_loss_select)
		model_num.set('model2')

"""
	if m==3:
		model_select = model_config_3
		image_model_select = Image.open(model_select.image_model)
		print(model_select.image_model)
		tk_model_select = ImageTk.PhotoImage(image_model_select)
"""
def select_model_1():
	select_mode(1)


def select_model_2():
	select_mode(2)


def select_model_3():
	select_mode(3)

def change_image():
	l.configure(image=tk_model_select)
	loss.configure(image=tk_loss_select)
	lr.configure(image=tk_lr_select)
def show_mode2_image():
	label_image_origin.configure(image=tk_image_change)
	#label_image_origin_optical.configure(image=image_change)
def show_strange():
	label_image_origin_optical.configure(image=tk_image_origin_optical_change)


def change_feature_map():
	label_image_layer1_1.configure(image=image_layer1_1_tk_84)
	label_image_layer1_2.configure(image=image_layer1_2_tk_84)
	label_image_layer2_1.configure(image=image_layer2_1_tk_84)
	label_image_layer2_2.configure(image=image_layer2_2_tk_84)
	label_image_layer3_1.configure(image=image_layer3_1_tk_84)
	label_image_layer3_2.configure(image=image_layer3_2_tk_84)
	label_image_layer4_1.configure(image=image_layer4_1_tk_84)
	label_image_layer4_2.configure(image=image_layer4_2_tk_84)
	label_image_layer5_1.configure(image=image_layer5_1_tk_84)
	label_image_layer5_2.configure(image=image_layer5_2_tk_84)
def show_feature_map():
	global image_layer1_1_tk_84
	global image_layer1_2_tk_84
	global image_layer2_1_tk_84
	global image_layer2_2_tk_84
	global image_layer3_1_tk_84
	global image_layer3_2_tk_84
	global image_layer4_1_tk_84
	global image_layer4_2_tk_84
	global image_layer5_1_tk_84
	global image_layer5_2_tk_84
	image_layer1_1_84 = Image.open('feature_map/layer1_1.jpg')
	image_layer1_1_tk_84 = ImageTk.PhotoImage(image_layer1_1_84)

	image_layer1_2_84 = Image.open('feature_map/layer1_2.jpg')
	image_layer1_2_tk_84 = ImageTk.PhotoImage(image_layer1_2_84)

	image_layer2_1_84 = Image.open('feature_map/layer2_1_84.jpg')
	image_layer2_1_tk_84 = ImageTk.PhotoImage(image_layer2_1_84)

	image_layer2_2_84 = Image.open('feature_map/layer2_2_84.jpg')
	image_layer2_2_tk_84 = ImageTk.PhotoImage(image_layer2_2_84)

	image_layer3_1_84 = Image.open('feature_map/layer3_1_84.jpg')
	image_layer3_1_tk_84 = ImageTk.PhotoImage(image_layer3_1_84)

	image_layer3_2_84 = Image.open('feature_map/layer3_2_84.jpg')
	image_layer3_2_tk_84 = ImageTk.PhotoImage(image_layer3_2_84)

	image_layer4_1_84 = Image.open('feature_map/layer4_1_84.jpg')
	image_layer4_1_tk_84 = ImageTk.PhotoImage(image_layer4_1_84)

	image_layer4_2_84 = Image.open('feature_map/layer4_2_84.jpg')
	image_layer4_2_tk_84 = ImageTk.PhotoImage(image_layer4_2_84)

	image_layer5_1_84 = Image.open('feature_map/layer5_1_84.jpg')
	image_layer5_1_tk_84 = ImageTk.PhotoImage(image_layer5_1_84)

	image_layer5_2_84 = Image.open('feature_map/layer5_2_84.jpg')
	image_layer5_2_tk_84 = ImageTk.PhotoImage(image_layer5_2_84)
	change_feature_map()
def run_1():
	evaluation_single(model_select.model_checkpoint_path,path_single)
	image_change = Image.open(path_single)
	global tk_image_change
	tk_image_change = ImageTk.PhotoImage(image_change)
	show_mode2_image()
	global text_image_label
	global text_image_label_output
	path_split = path_single.split('/')
	text_image_label.set(array_class[int(path_split[-2])])
	text_image_label_output.set(array_class[int(index_predict)])
	image_change_optical = Image.open('Optical_image/%s.png'%(array_class[int(path_split[-2])]))
	global tk_image_origin_optical_change
	tk_image_origin_optical_change = ImageTk.PhotoImage(image_change_optical)
	text2 = Text(frame2,width=15,height=8)
	show_feature_map()
	if text_image_label.get()==text_image_label_output.get():
		text2.insert(0.0,'检测正确\n请继续使用')
		show_strange()
		#text2.place(x=700,y=60)

	if text_image_label.get() != text_image_label_output.get():
		messagebox.showwarning('警告','检测错误,请选择其他模型')

def run_2():
	path_record_split = path_record.split('.')
	print(path_record_split[-1])
	if path_record_split[-1] == "tfrecords":
		evaluation(model_select.model_checkpoint_path,path_record)
		result_array = final_results
		result_00.set(int(result_array[0][0])), result_01.set(int(result_array[0][1])), result_02.set(int(result_array[0][2])), result_03.set(int(result_array[0][3])), result_04.set(int(result_array[0][4])), result_05.set(int(result_array[0][5])), result_06.set(int(result_array[0][6])), result_07.set(int(result_array[0][7])), result_08.set(int(result_array[0][8])), result_09.set(int(result_array[0][9]))
		result_10.set(int(result_array[1][0])), result_11.set(int(result_array[1][1])), result_12.set(int(result_array[1][2])), result_13.set(int(result_array[1][3])), result_14.set(int(result_array[1][4])), result_15.set(int(result_array[1][5])), result_16.set(int(result_array[1][6])), result_17.set(int(result_array[1][7])), result_18.set(int(result_array[1][8])), result_19.set(int(result_array[1][9]))
		result_20.set(int(result_array[2][0])), result_21.set(int(result_array[2][1])), result_22.set(int(result_array[2][2])), result_23.set(int(result_array[2][3])), result_24.set(int(result_array[2][4])), result_25.set(int(result_array[2][5])), result_26.set(int(result_array[2][6])), result_27.set(int(result_array[2][7])), result_28.set(int(result_array[2][8])), result_29.set(int(result_array[2][9]))
		result_30.set(int(result_array[3][0])), result_31.set(int(result_array[3][1])), result_32.set(int(result_array[3][2])), result_33.set(int(result_array[3][3])), result_34.set(int(result_array[3][4])), result_35.set(int(result_array[3][5])), result_36.set(int(result_array[3][6])), result_37.set(int(result_array[3][7])), result_38.set(int(result_array[3][8])), result_39.set(int(result_array[3][9]))
		result_40.set(int(result_array[4][0])), result_41.set(int(result_array[4][1])), result_42.set(int(result_array[4][2])), result_43.set(int(result_array[4][3])), result_44.set(int(result_array[4][4])), result_45.set(int(result_array[4][5])), result_46.set(int(result_array[4][6])), result_47.set(int(result_array[4][7])), result_48.set(int(result_array[4][8])), result_49.set(int(result_array[4][9]))
		result_50.set(int(result_array[5][0])), result_51.set(int(result_array[5][1])), result_52.set(int(result_array[5][2])), result_53.set(int(result_array[5][3])), result_54.set(int(result_array[5][4])), result_55.set(int(result_array[5][5])), result_56.set(int(result_array[5][6])), result_57.set(int(result_array[5][7])), result_58.set(int(result_array[5][8])), result_59.set(int(result_array[5][9]))
		result_60.set(int(result_array[6][0])), result_61.set(int(result_array[6][1])), result_62.set(int(result_array[6][2])), result_63.set(int(result_array[6][3])), result_64.set(int(result_array[6][4])), result_65.set(int(result_array[6][5])), result_66.set(int(result_array[6][6])), result_67.set(int(result_array[6][7])), result_68.set(int(result_array[6][8])), result_69.set(int(result_array[6][9]))
		result_70.set(int(result_array[7][0])), result_71.set(int(result_array[7][1])), result_72.set(int(result_array[7][2])), result_73.set(int(result_array[7][3])), result_74.set(int(result_array[7][4])), result_75.set(int(result_array[7][5])), result_76.set(int(result_array[7][6])), result_77.set(int(result_array[7][7])), result_78.set(int(result_array[7][8])), result_79.set(int(result_array[7][9]))
		result_80.set(int(result_array[8][0])), result_81.set(int(result_array[8][1])), result_82.set(int(result_array[8][2])), result_83.set(int(result_array[8][3])), result_84.set(int(result_array[8][4])), result_85.set(int(result_array[8][5])), result_86.set(int(result_array[8][6])), result_87.set(int(result_array[8][7])), result_88.set(int(result_array[8][8])), result_89.set(int(result_array[8][9]))
		result_90.set(int(result_array[9][0])), result_91.set(int(result_array[9][1])), result_92.set(int(result_array[9][2])), result_93.set(int(result_array[9][3])), result_94.set(int(result_array[9][4])), result_95.set(int(result_array[9][5])), result_96.set(int(result_array[9][6])), result_97.set(int(result_array[9][7])), result_98.set(int(result_array[9][8])), result_99.set(int(result_array[9][9]))
	if path_record_split[-1] != "tfrecords":
		messagebox.showwarning('错误','请选择tfrecords文件')


#### define frame1###
#Button(frame1,text="button1",background="LightSeaGreen",fg='Red').place(x=1000,y=550)
#Button(frame1,text="button2",bg="PeachPuff").pack()
image_background = Image.open('SAR/model/image1.jpg')
tk_background = ImageTk.PhotoImage(image_background)
background = Label(frame1,image=tk_background)
background.pack()
#Label(frame1,text='Radar Detection System Based',font=("Courier New",32)).place(x=250,y=200)
#Label(frame1,text='On Depth Learing',font=("Courier New",32)).place(x=340,y=270)
#Label(frame1,text='北航新主楼F523',font=("Courier New",20)).place(x=650,y=400)
#Label(frame1,text=teacher,font=("Courier New",20)).place(x=650,y=480)
#Button(frame1,text='退出',command=exit).place(x=950,y=550)
#background.place(x=0,y=0)
#### define frame3###
Label(frame3,text='模型结构',bg='LightSkyBlue',font=("Arial",11)).place(x=20,y=20)
Label(frame3,text='损失函数',bg='LightSkyBlue',font=("Arial",11)).place(x=20,y=330)
Label(frame3,text='学习率',bg='LightSkyBlue',font=("Arial",11)).place(x=450,y=330)
Button(frame3,text='退出',command=exit).place(x=950,y=550)
image_model_1 = Image.open(model_config_1.image_model)
tk_model = ImageTk.PhotoImage(image_model_1)
l = Label(frame3,image=tk_model)
l.place(x=50,y=50)
image_lr_1 = Image.open(model_config_1.image_lr)
tk_lr = ImageTk.PhotoImage(image_lr_1)
lr = Label(frame3,image=tk_lr)
lr.place(x=50,y=360)
image_loss_1 = Image.open(model_config_1.image_loss)
tk_loss = ImageTk.PhotoImage(image_loss_1)
loss = Label(frame3,image=tk_loss)
loss.place(x=480,y=360)
#intro = ScrolledText(frame3,width=20,height=10,font=("Arial",12))
#intro.place(x=850,y=200)
button_refresh = Button(frame3,text="update",command=change_image)
button_refresh.place(x=950,y=50)
#### define frame2###
path = StringVar()

def selectPath():
	path_ = askopenfilename()
	path.set(path_)
	global path_single
	path_single=path_

	#print(path_single)
def exit():
    root.destroy()
Label(frame2,text = "目标路径:",font=("Arial",10),bg='LightSkyBlue',fg='Black').place(x=350,y=12)
entry1 = Entry(frame2, textvariable = path,font=("Arial",10)).place(x=430,y=12)
Button(frame2, text = "路径选择", command = selectPath,font=("Arial",10),bg='LightSkyBlue',fg='Black').place(x=580,y=10)
text1 = Text(frame2,width=15,height=18)
#text1.insert(0.0,'成功')
#text1.place(x=700,y=60)
Button(frame2, text = "退出", command = exit,font=("Arial",10),bg='LightSkyBlue',fg='Black').place(x=950,y=550)
Button(frame2, text = "运行", command = run_1,font=("Arial",10),bg='LightSkyBlue',fg='Black').place(x=670,y=10)
#Button(frame2, text = "显示特征图", command = show_feature_map,font=("Arial",10),bg='LightSkyBlue',fg='Black').place(x=470,y=200)
Label(frame2,text = "第一层",font=("Arial",11),bg='LightSkyBlue',fg='Black').place(x=270,y=270)
Label(frame2,text = "第二层",font=("Arial",11),bg='LightSkyBlue',fg='Black').place(x=400,y=270)
Label(frame2,text = "第三层",font=("Arial",11),bg='LightSkyBlue',fg='Black').place(x=530,y=270)
Label(frame2,text = "第四层",font=("Arial",11),bg='LightSkyBlue',fg='Black').place(x=660,y=270)
Label(frame2,text = "第五层",font=("Arial",11),bg='LightSkyBlue',fg='Black').place(x=790,y=270)
Label(frame2,text = "特",font=("Arial",11),bg='LightSkyBlue',fg='Black').place(x=200,y=355)
Label(frame2,text = "征",font=("Arial",11),bg='LightSkyBlue',fg='Black').place(x=200,y=395)
Label(frame2,text = "图",font=("Arial",11),bg='LightSkyBlue',fg='Black').place(x=200,y=435)

image_origin = Image.open(path_single)
tk_image_origin = ImageTk.PhotoImage(image_origin)
label_image_origin = Label(frame2,image=tk_image_origin)
label_image_origin.place(x=300,y=80)


image_layer1_1 = Image.open('feature_map/layer1_1.jpg')
image_layer1_1_tk = ImageTk.PhotoImage(image_layer1_1)
label_image_layer1_1 = Label(frame2,image=image_layer1_1_tk)
label_image_layer1_1.place(x=250,y=300)
print("label_image_layer1_1")

image_layer1_2 = Image.open('feature_map/layer1_2.jpg')
image_layer1_2_tk = ImageTk.PhotoImage(image_layer1_2)
label_image_layer1_2 = Label(frame2,image=image_layer1_2_tk)
label_image_layer1_2.place(x=250,y=430)

image_layer2_1 = Image.open('feature_map/layer2_1_84.jpg')
image_layer2_1_tk = ImageTk.PhotoImage(image_layer2_1)
label_image_layer2_1 = Label(frame2,image=image_layer2_1_tk)
label_image_layer2_1.place(x=380,y=300)

image_layer2_2 = Image.open('feature_map/layer2_2_84.jpg')
image_layer2_2_tk = ImageTk.PhotoImage(image_layer2_2)
label_image_layer2_2 = Label(frame2,image=image_layer2_2_tk)
label_image_layer2_2.place(x=380,y=430)

image_layer3_1 = Image.open('feature_map/layer3_1_84.jpg')
image_layer3_1_tk = ImageTk.PhotoImage(image_layer3_1)
label_image_layer3_1 = Label(frame2,image=image_layer3_1_tk)
label_image_layer3_1.place(x=510,y=300)

image_layer3_2 = Image.open('feature_map/layer3_2_84.jpg')
image_layer3_2_tk = ImageTk.PhotoImage(image_layer3_2)
label_image_layer3_2 = Label(frame2,image=image_layer3_2_tk)
label_image_layer3_2.place(x=510,y=430)

image_layer4_1 = Image.open('feature_map/layer4_1_84.jpg')
image_layer4_1_tk = ImageTk.PhotoImage(image_layer4_1)
label_image_layer4_1 = Label(frame2,image=image_layer4_1_tk)
label_image_layer4_1.place(x=640,y=300)

image_layer4_2 = Image.open('feature_map/layer4_2_84.jpg')
image_layer4_2_tk = ImageTk.PhotoImage(image_layer4_2)
label_image_layer4_2 = Label(frame2,image=image_layer4_2_tk)
label_image_layer4_2.place(x=640,y=430)

image_layer5_1 = Image.open('feature_map/layer5_1_84.jpg')
image_layer5_1_tk = ImageTk.PhotoImage(image_layer5_1)
label_image_layer5_1 = Label(frame2,image=image_layer5_1_tk)
label_image_layer5_1.place(x=770,y=300)

image_layer5_2 = Image.open('feature_map/layer5_2_84.jpg')
image_layer5_2_tk = ImageTk.PhotoImage(image_layer5_2)
label_image_layer5_2 = Label(frame2,image=image_layer5_2_tk)
label_image_layer5_2.place(x=770,y=430)



text_image_label_output = StringVar()
text_image_label = StringVar()
model_num = StringVar()
path_split = path_single.split('/')
text_image_label.set(array_class[int(path_split[-2])])

image_origin_optical = Image.open('Optical_image/%s.png'%(array_class[int(path_split[-2])]))
tk_image_origin_optical = ImageTk.PhotoImage(image_origin_optical)
label_image_origin_optical = Label(frame2,image=tk_image_origin_optical)
label_image_origin_optical.place(x=700,y=75)

print(index_predict)
text_image_label_output.set(index_predict)
Label(frame2,text = "原始图像:",font=("Arial",11),bg='LightSkyBlue',fg='Black').place(x=200,y=90)
Label(frame2,text = "真实类别:",font=("Arial",11),bg='LightSkyBlue',fg='Black').place(x=450,y=90)
Label(frame2,textvariable=text_image_label,font=("Arial",11),bg='LightSkyBlue').place(x=600,y=90)
Label(frame2,text = "预测类别:",font=("Arial",11),bg='LightSkyBlue',fg='Red').place(x=450,y=140)
Label(frame2,textvariable=text_image_label_output,font=("Arial",11),bg='LightSkyBlue',fg='Red').place(x=600,y=140)

Label(frame2,textvariable=model_num,font=("Arial",11),bg='LightSkyBlue').place(x=950,y=20)
Label(frame3,textvariable=model_num,font=("Arial",11),bg='LightSkyBlue').place(x=950,y=20)
Label(frame4,textvariable=model_num,font=("Arial",11),bg='LightSkyBlue').place(x=950,y=20)
#### define frame4###

path_frame4 = StringVar()

result_00 = StringVar()
result_01 = StringVar()
result_02 = StringVar()
result_03 = StringVar()
result_04 = StringVar()
result_05 = StringVar()
result_06 = StringVar()
result_07 = StringVar()
result_08 = StringVar()
result_09 = StringVar()
result_10 = StringVar()
result_11 = StringVar()
result_12 = StringVar()
result_13 = StringVar()
result_14 = StringVar()
result_15 = StringVar()
result_16 = StringVar()
result_17 = StringVar()
result_18 = StringVar()
result_19 = StringVar()
result_20 = StringVar()
result_21 = StringVar()
result_22 = StringVar()
result_23 = StringVar()
result_24 = StringVar()
result_25 = StringVar()
result_26 = StringVar()
result_27 = StringVar()
result_28 = StringVar()
result_29 = StringVar()
result_30 = StringVar()
result_31 = StringVar()
result_32 = StringVar()
result_33 = StringVar()
result_34 = StringVar()
result_35 = StringVar()
result_36 = StringVar()
result_37 = StringVar()
result_38 = StringVar()
result_39 = StringVar()
result_40 = StringVar()
result_41 = StringVar()
result_42 = StringVar()
result_43 = StringVar()
result_44 = StringVar()
result_45 = StringVar()
result_46 = StringVar()
result_47 = StringVar()
result_48 = StringVar()
result_49 = StringVar()
result_50 = StringVar()
result_51 = StringVar()
result_52 = StringVar()
result_53 = StringVar()
result_54 = StringVar()
result_55 = StringVar()
result_56 = StringVar()
result_57 = StringVar()
result_58 = StringVar()
result_59 = StringVar()
result_60 = StringVar()
result_61 = StringVar()
result_62 = StringVar()
result_63 = StringVar()
result_64 = StringVar()
result_65 = StringVar()
result_66 = StringVar()
result_67 = StringVar()
result_68 = StringVar()
result_69 = StringVar()
result_70 = StringVar()
result_71 = StringVar()
result_72 = StringVar()
result_73 = StringVar()
result_74 = StringVar()
result_75 = StringVar()
result_76 = StringVar()
result_77 = StringVar()
result_78 = StringVar()
result_79 = StringVar()
result_80 = StringVar()
result_81 = StringVar()
result_82 = StringVar()
result_83 = StringVar()
result_84 = StringVar()
result_85 = StringVar()
result_86 = StringVar()
result_87 = StringVar()
result_88 = StringVar()
result_89 = StringVar()
result_90 = StringVar()
result_91 = StringVar()
result_92 = StringVar()
result_93 = StringVar()
result_94 = StringVar()
result_95 = StringVar()
result_96 = StringVar()
result_97 = StringVar()
result_98 = StringVar()
result_99 = StringVar()
Label(frame4,textvariable=result_00,font=("Arial",11),bg='LightSkyBlue').place(x=300,y=95),Label(frame4,textvariable=result_01,font=("Arial",11),bg='LightSkyBlue').place(x=350,y=95)
Label(frame4,textvariable=result_02,font=("Arial",11),bg='LightSkyBlue').place(x=400,y=95),Label(frame4,textvariable=result_03,font=("Arial",11),bg='LightSkyBlue').place(x=465,y=95)
Label(frame4,textvariable=result_04,font=("Arial",11),bg='LightSkyBlue').place(x=530,y=95),Label(frame4,textvariable=result_05,font=("Arial",11),bg='LightSkyBlue').place(x=600,y=95)
Label(frame4,textvariable=result_06,font=("Arial",11),bg='LightSkyBlue').place(x=650,y=95),Label(frame4,textvariable=result_07,font=("Arial",11),bg='LightSkyBlue').place(x=700,y=95)
Label(frame4,textvariable=result_08,font=("Arial",11),bg='LightSkyBlue').place(x=750,y=95),Label(frame4,textvariable=result_09,font=("Arial",11),bg='LightSkyBlue').place(x=820,y=95)

Label(frame4,textvariable=result_10,font=("Arial",11),bg='LightSkyBlue').place(x=300,y=120),Label(frame4,textvariable=result_11,font=("Arial",11),bg='LightSkyBlue').place(x=350,y=120)
Label(frame4,textvariable=result_12,font=("Arial",11),bg='LightSkyBlue').place(x=400,y=120),Label(frame4,textvariable=result_13,font=("Arial",11),bg='LightSkyBlue').place(x=465,y=120)
Label(frame4,textvariable=result_14,font=("Arial",11),bg='LightSkyBlue').place(x=530,y=120),Label(frame4,textvariable=result_15,font=("Arial",11),bg='LightSkyBlue').place(x=600,y=120)
Label(frame4,textvariable=result_16,font=("Arial",11),bg='LightSkyBlue').place(x=650,y=120),Label(frame4,textvariable=result_17,font=("Arial",11),bg='LightSkyBlue').place(x=700,y=120)
Label(frame4,textvariable=result_18,font=("Arial",11),bg='LightSkyBlue').place(x=750,y=120),Label(frame4,textvariable=result_19,font=("Arial",11),bg='LightSkyBlue').place(x=820,y=120)

Label(frame4,textvariable=result_20,font=("Arial",11),bg='LightSkyBlue').place(x=300,y=145),Label(frame4,textvariable=result_21,font=("Arial",11),bg='LightSkyBlue').place(x=350,y=145)
Label(frame4,textvariable=result_22,font=("Arial",11),bg='LightSkyBlue').place(x=400,y=145),Label(frame4,textvariable=result_23,font=("Arial",11),bg='LightSkyBlue').place(x=465,y=145)
Label(frame4,textvariable=result_24,font=("Arial",11),bg='LightSkyBlue').place(x=530,y=145),Label(frame4,textvariable=result_25,font=("Arial",11),bg='LightSkyBlue').place(x=600,y=145)
Label(frame4,textvariable=result_26,font=("Arial",11),bg='LightSkyBlue').place(x=650,y=145),Label(frame4,textvariable=result_27,font=("Arial",11),bg='LightSkyBlue').place(x=700,y=145)
Label(frame4,textvariable=result_28,font=("Arial",11),bg='LightSkyBlue').place(x=750,y=145),Label(frame4,textvariable=result_29,font=("Arial",11),bg='LightSkyBlue').place(x=820,y=145)

Label(frame4,textvariable=result_30,font=("Arial",11),bg='LightSkyBlue').place(x=300,y=170),Label(frame4,textvariable=result_31,font=("Arial",11),bg='LightSkyBlue').place(x=350,y=170)
Label(frame4,textvariable=result_32,font=("Arial",11),bg='LightSkyBlue').place(x=400,y=170),Label(frame4,textvariable=result_33,font=("Arial",11),bg='LightSkyBlue').place(x=465,y=170)
Label(frame4,textvariable=result_34,font=("Arial",11),bg='LightSkyBlue').place(x=530,y=170),Label(frame4,textvariable=result_35,font=("Arial",11),bg='LightSkyBlue').place(x=600,y=170)
Label(frame4,textvariable=result_36,font=("Arial",11),bg='LightSkyBlue').place(x=650,y=170),Label(frame4,textvariable=result_37,font=("Arial",11),bg='LightSkyBlue').place(x=700,y=170)
Label(frame4,textvariable=result_38,font=("Arial",11),bg='LightSkyBlue').place(x=750,y=170),Label(frame4,textvariable=result_39,font=("Arial",11),bg='LightSkyBlue').place(x=820,y=170)

Label(frame4,textvariable=result_40,font=("Arial",11),bg='LightSkyBlue').place(x=300,y=195),Label(frame4,textvariable=result_41,font=("Arial",11),bg='LightSkyBlue').place(x=350,y=195)
Label(frame4,textvariable=result_42,font=("Arial",11),bg='LightSkyBlue').place(x=400,y=195),Label(frame4,textvariable=result_43,font=("Arial",11),bg='LightSkyBlue').place(x=465,y=195)
Label(frame4,textvariable=result_44,font=("Arial",11),bg='LightSkyBlue').place(x=530,y=195),Label(frame4,textvariable=result_45,font=("Arial",11),bg='LightSkyBlue').place(x=600,y=195)
Label(frame4,textvariable=result_46,font=("Arial",11),bg='LightSkyBlue').place(x=650,y=195),Label(frame4,textvariable=result_47,font=("Arial",11),bg='LightSkyBlue').place(x=700,y=195)
Label(frame4,textvariable=result_48,font=("Arial",11),bg='LightSkyBlue').place(x=750,y=195),Label(frame4,textvariable=result_49,font=("Arial",11),bg='LightSkyBlue').place(x=820,y=195)

Label(frame4,textvariable=result_50,font=("Arial",11),bg='LightSkyBlue').place(x=300,y=220),Label(frame4,textvariable=result_51,font=("Arial",11),bg='LightSkyBlue').place(x=350,y=220)
Label(frame4,textvariable=result_52,font=("Arial",11),bg='LightSkyBlue').place(x=400,y=220),Label(frame4,textvariable=result_53,font=("Arial",11),bg='LightSkyBlue').place(x=465,y=220)
Label(frame4,textvariable=result_54,font=("Arial",11),bg='LightSkyBlue').place(x=530,y=220),Label(frame4,textvariable=result_55,font=("Arial",11),bg='LightSkyBlue').place(x=600,y=220)
Label(frame4,textvariable=result_56,font=("Arial",11),bg='LightSkyBlue').place(x=650,y=220),Label(frame4,textvariable=result_57,font=("Arial",11),bg='LightSkyBlue').place(x=700,y=220)
Label(frame4,textvariable=result_58,font=("Arial",11),bg='LightSkyBlue').place(x=750,y=220),Label(frame4,textvariable=result_59,font=("Arial",11),bg='LightSkyBlue').place(x=820,y=220)

Label(frame4,textvariable=result_60,font=("Arial",11),bg='LightSkyBlue').place(x=300,y=245),Label(frame4,textvariable=result_61,font=("Arial",11),bg='LightSkyBlue').place(x=350,y=245)
Label(frame4,textvariable=result_62,font=("Arial",11),bg='LightSkyBlue').place(x=400,y=245),Label(frame4,textvariable=result_63,font=("Arial",11),bg='LightSkyBlue').place(x=465,y=245)
Label(frame4,textvariable=result_64,font=("Arial",11),bg='LightSkyBlue').place(x=530,y=245),Label(frame4,textvariable=result_65,font=("Arial",11),bg='LightSkyBlue').place(x=600,y=245)
Label(frame4,textvariable=result_66,font=("Arial",11),bg='LightSkyBlue').place(x=650,y=245),Label(frame4,textvariable=result_67,font=("Arial",11),bg='LightSkyBlue').place(x=700,y=245)
Label(frame4,textvariable=result_68,font=("Arial",11),bg='LightSkyBlue').place(x=750,y=245),Label(frame4,textvariable=result_69,font=("Arial",11),bg='LightSkyBlue').place(x=820,y=245)

Label(frame4,textvariable=result_70,font=("Arial",11),bg='LightSkyBlue').place(x=300,y=270),Label(frame4,textvariable=result_71,font=("Arial",11),bg='LightSkyBlue').place(x=350,y=270)
Label(frame4,textvariable=result_72,font=("Arial",11),bg='LightSkyBlue').place(x=400,y=270),Label(frame4,textvariable=result_73,font=("Arial",11),bg='LightSkyBlue').place(x=465,y=270)
Label(frame4,textvariable=result_74,font=("Arial",11),bg='LightSkyBlue').place(x=530,y=270),Label(frame4,textvariable=result_75,font=("Arial",11),bg='LightSkyBlue').place(x=600,y=270)
Label(frame4,textvariable=result_76,font=("Arial",11),bg='LightSkyBlue').place(x=650,y=270),Label(frame4,textvariable=result_77,font=("Arial",11),bg='LightSkyBlue').place(x=700,y=270)
Label(frame4,textvariable=result_78,font=("Arial",11),bg='LightSkyBlue').place(x=750,y=270),Label(frame4,textvariable=result_79,font=("Arial",11),bg='LightSkyBlue').place(x=820,y=270)

Label(frame4,textvariable=result_80,font=("Arial",11),bg='LightSkyBlue').place(x=300,y=295),Label(frame4,textvariable=result_81,font=("Arial",11),bg='LightSkyBlue').place(x=350,y=295)
Label(frame4,textvariable=result_82,font=("Arial",11),bg='LightSkyBlue').place(x=400,y=295),Label(frame4,textvariable=result_83,font=("Arial",11),bg='LightSkyBlue').place(x=465,y=295)
Label(frame4,textvariable=result_84,font=("Arial",11),bg='LightSkyBlue').place(x=530,y=295),Label(frame4,textvariable=result_85,font=("Arial",11),bg='LightSkyBlue').place(x=600,y=295)
Label(frame4,textvariable=result_86,font=("Arial",11),bg='LightSkyBlue').place(x=650,y=295),Label(frame4,textvariable=result_87,font=("Arial",11),bg='LightSkyBlue').place(x=700,y=295)
Label(frame4,textvariable=result_88,font=("Arial",11),bg='LightSkyBlue').place(x=750,y=295),Label(frame4,textvariable=result_89,font=("Arial",11),bg='LightSkyBlue').place(x=820,y=295)

Label(frame4,textvariable=result_90,font=("Arial",11),bg='LightSkyBlue').place(x=300,y=320),Label(frame4,textvariable=result_91,font=("Arial",11),bg='LightSkyBlue').place(x=350,y=320)
Label(frame4,textvariable=result_92,font=("Arial",11),bg='LightSkyBlue').place(x=400,y=320),Label(frame4,textvariable=result_93,font=("Arial",11),bg='LightSkyBlue').place(x=465,y=320)
Label(frame4,textvariable=result_94,font=("Arial",11),bg='LightSkyBlue').place(x=530,y=320),Label(frame4,textvariable=result_95,font=("Arial",11),bg='LightSkyBlue').place(x=600,y=320)
Label(frame4,textvariable=result_96,font=("Arial",11),bg='LightSkyBlue').place(x=650,y=320),Label(frame4,textvariable=result_97,font=("Arial",11),bg='LightSkyBlue').place(x=700,y=320)
Label(frame4,textvariable=result_98,font=("Arial",11),bg='LightSkyBlue').place(x=750,y=320),Label(frame4,textvariable=result_99,font=("Arial",11),bg='LightSkyBlue').place(x=820,y=320)

def selectPath_frame4():
	path_ = askopenfilename()
	path.set(path_)
	global path_record
	path_record=path_
	print(path_record)
Button(frame4, text = "退出", command = exit,font=("Arial",10),bg='LightSkyBlue',fg='Black').place(x=950,y=550)
Label(frame4,text = "目标路径:",font=("Arial",10),bg='LightSkyBlue',fg='Black').place(x=350,y=12)
entry1 = Entry(frame4, textvariable = path,font=("Arial",10)).place(x=430,y=12)
Button(frame4, text = "路径选择", command = selectPath_frame4,font=("Arial",10),bg='LightSkyBlue',fg='Black').place(x=580,y=10)
Button(frame4, text = "运行", command = run_2,font=("Arial",10),bg='LightSkyBlue',fg='Black').place(x=670,y=10)
Label(frame4,text = "原",font=("Arial",11),bg='LightSkyBlue',fg='Black').place(x=160,y=170)
Label(frame4,text = "始",font=("Arial",11),bg='LightSkyBlue',fg='Black').place(x=160,y=190)
Label(frame4,text = "类",font=("Arial",11),bg='LightSkyBlue',fg='Black').place(x=160,y=210)
Label(frame4,text = "别",font=("Arial",11),bg='LightSkyBlue',fg='Black').place(x=160,y=230)
Label(frame4,text = "(274)2S1",font=("Arial",11),bg='LightSkyBlue',fg='Black').place(x=190,y=95)
Label(frame4,text = "(195)BMP2",font=("Arial",11),bg='LightSkyBlue',fg='Black').place(x=190,y=120)
Label(frame4,text = "(274)BRDM2",font=("Arial",11),bg='LightSkyBlue',fg='Black').place(x=190,y=145)
Label(frame4,text = "(195)BTR60",font=("Arial",11),bg='LightSkyBlue',fg='Black').place(x=190,y=170)
Label(frame4,text = "(196)BTR70",font=("Arial",11),bg='LightSkyBlue',fg='Black').place(x=190,y=195)
Label(frame4,text = "(274)D7",font=("Arial",11),bg='LightSkyBlue',fg='Black').place(x=190,y=220)
Label(frame4,text = "(273)T62",font=("Arial",11),bg='LightSkyBlue',fg='Black').place(x=190,y=245)
Label(frame4,text = "(196)T72",font=("Arial",11),bg='LightSkyBlue',fg='Black').place(x=190,y=270)
Label(frame4,text = "(274)ZIL131",font=("Arial",11),bg='LightSkyBlue',fg='Black').place(x=190,y=295)
Label(frame4,text = "(274)ZSU234",font=("Arial",11),bg='LightSkyBlue',fg='Black').place(x=190,y=320)
Label(frame4,text = "预 测 类 别:",font=("Arial",11),bg='LightSkyBlue',fg='Black').place(x=500,y=50)
Label(frame4,text = "2S1 ",font=("Arial",11),bg='LightSkyBlue',fg='Black').place(x=300,y=70)
Label(frame4,text = "BMP2 ",font=("Arial",11),bg='LightSkyBlue',fg='Black').place(x=350,y=70)
Label(frame4,text = "BRDM2 ",font=("Arial",11),bg='LightSkyBlue',fg='Black').place(x=400,y=70)
Label(frame4,text = "BTR60 ",font=("Arial",11),bg='LightSkyBlue',fg='Black').place(x=465,y=70)
Label(frame4,text = "BTR70 ",font=("Arial",11),bg='LightSkyBlue',fg='Black').place(x=530,y=70)
Label(frame4,text = "D7 ",font=("Arial",11),bg='LightSkyBlue',fg='Black').place(x=600,y=70)
Label(frame4,text = "T62 ",font=("Arial",11),bg='LightSkyBlue',fg='Black').place(x=650,y=70)
Label(frame4,text = "T72 ",font=("Arial",11),bg='LightSkyBlue',fg='Black').place(x=700,y=70)
Label(frame4,text = "ZIL131 ",font=("Arial",11),bg='LightSkyBlue',fg='Black').place(x=750,y=70)
Label(frame4,text = "ZSU234 ",font=("Arial",11),bg='LightSkyBlue',fg='Black').place(x=820,y=70)
#array_class = ['2S1','BMP2','BRDM2','BTR60','BTR70','D7','T62','T72','ZIL131','ZSU234']
scrotext = ScrolledText(frame4,width=63,height=13,font=("Arial",12))
scrotext.place(x=250,y=360)
#[274, 195, 274, 195, 196, 274, 273, 196, 274, 274]
menubar=Menu(root)
function_select_menu = Menu(menubar,tearoff=0)
function_select_menu.add_command(label="主界面",command=to_frame1)
function_select_menu.add_command(label="模型信息",command=to_frame3)
function_select_menu.add_command(label="单幅图像测试",command=to_frame2)
function_select_menu.add_command(label="数据集测试",command=to_frame4)
function_select_model = Menu(menubar,tearoff=0)
function_select_model.add_command(label="模型1",command=select_model_1)
function_select_model.add_command(label="模型2",command=select_model_2)
#function_select_model.add_command(label="model3",command=select_model_3)
menubar.add_cascade(label="功能",menu=function_select_menu)
menubar.add_cascade(label="模型",menu=function_select_model)
root.config(menu=menubar)

root.mainloop()

#def main(argv=None):
#    pass
    #evaluation()
    #evaluation_single()


#if __name__ == '__main__':
#    tf.app.run()
