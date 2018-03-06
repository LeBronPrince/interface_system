import tensorflow as tf
from PIL import Image
import numpy as np
'''
注释部分为测试代码，测试生成的batch

IMAGE_SIZE = 88  # 输入图像大小为88*88
BATCH_SIZE = 128  # batch的大小是128

readpath = '/home/f523/LJY/SAR/MSTAR/train/Label'
filename = readpath + '/train.tfrecords'

filename_queue = tf.train.string_input_producer([filename], shuffle=False)  # 生成文件队列

reader = tf.TFRecordReader()  # 创建一个TFRecords文件阅读器
_, serialized_example = reader.read(filename_queue)  # 读取队列,得到序列化的实例

features = tf.parse_single_example(serialized_example,
                                   features={
                                       'label': tf.FixedLenFeature([], tf.int64),
                                       'image_raw': tf.FixedLenFeature([], tf.string)
                                   })

image = tf.decode_raw(features['image_raw'], tf.uint8)
image = tf.reshape(image, [IMAGE_SIZE, IMAGE_SIZE, 1])  # 3-D tensor with shape [88, 88, 1]
image = tf.cast(image, tf.float32)  # Convert uint8 to float32

label = tf.cast(features['label'], tf.int32)  # 1-D tensor of int32

image_batch, label_batch = tf.train.batch([image, label],
                                          batch_size = BATCH_SIZE,
                                          capacity = 1000 + 3*BATCH_SIZE)  # 生成图像的batch和标签的batch

# 在session中，若执行语句cur_image_batch, cur_label_batch = sess.run([image_batch, label_batch]，
# 得到的cur_image_batch的shape是(BATCH_SIZE, 88, 88, 1),cur_label_batch的shape是(BATCH_SIZE,)
with tf.Session() as sess:
    init=tf.global_variables_initializer()
    sess.run(init)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    cur_image_batch, cur_label_batch = sess.run([image_batch, label_batch])
    cur_image_batch = cur_image_batch / 255.  # 将图像灰度值归一化到[0,1]之间
    print(cur_image_batch)
    print(cur_image_batch.shape)
    print('-------------')
    print(cur_label_batch)
    print(cur_label_batch.shape)

    coord.request_stop()
    coord.join(threads)
'''

def read_single_image(filename, image_size):
    image = Image.open(filename)
    image = tf.reshape(np.array(image), [image_size, image_size, 1])  # 3-D tensor with shape [88, 88, 1]
    image = tf.cast(image, tf.float32)
    #label = tf.constant([2])
    label = 2
    label = tf.cast(label, tf.int32)
    print(label.shape)
    return image,label

def read_data_sets(filename, image_size):
    '''
    该函数实现从TFRecords文件中读取数据
    :param filename: TFRecords文件名
    :param image_size: 生成的图像大小
    :return: image：type tf.float32, shape [image_size, image_size, 1]
             label: type tf.int32
    '''

    filename_queue = tf.train.string_input_producer([filename], shuffle=False)  # 生成文件队列

    reader = tf.TFRecordReader()  # 创建一个TFRecords文件阅读器
    _, serialized_example = reader.read(filename_queue)  # 读取队列,得到序列化的实例

    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'image_raw': tf.FixedLenFeature([], tf.string)
                                       })

    image = tf.decode_raw(features['image_raw'], tf.uint8)
    image = tf.reshape(image, [image_size, image_size, 1])  # 3-D tensor with shape [88, 88, 1]
    image = tf.cast(image, tf.float32)  # Convert uint8 to float32
    #print(image)
    label = tf.cast(features['label'], tf.int32)  # 1-D tensor of int32
    print(label.shape)
    return image, label


def generate_batch(image, label, batch_size, shuffle=None):
    '''
    生成图像(image)和标签(label)的batch...在会话Session中执行,采用多线程.
    :param image: read_data_sets生成的image
    :param label: read_data_sets生成的label
    :param batch_size: 组合成batch的大小
    :param shuffle: 组合batch时是否进行shuffle,shuffle=None时不执行shuffle
    :return: image_batch： 生成的图像batch
             label_batch： 生成的标签batch
    '''
    if shuffle is not None:
        image_batch, label_batch = tf.train.shuffle_batch([image, label],
                                                          batch_size=batch_size,
                                                          capacity=50000,
                                                          min_after_dequeue=10000,
                                                          num_threads=4)
    image_batch, label_batch = tf.train.batch([image, label],
                                              batch_size=batch_size,
                                              capacity=1000+3*batch_size)

    return image_batch, label_batch
