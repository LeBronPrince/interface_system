import os
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt


# Generate Int Type Feature
def _int64_feature(value):
    return tf.train.Feature(int64_list = tf.train.Int64List(value=[value]))

# Generate Byte Type Feature
def _bytes_feature(value):
    return tf.train.Feature(bytes_list = tf.train.BytesList(value=[value]))

# Generate Float Type Feature
def _float_feature(value):
    return tf.train.Feature(_float_feature = tf.train.FloatList(value=[value]))


readpath = '/home/wangyang/下载/SAR/test/Label'
filename = readpath + '/label_9.txt'
recordname = readpath + '/test_9.tfrecords'

writer = tf.python_io.TFRecordWriter(recordname)

f = open(filename,'r')

while True:
    line = f.readline()
    if not line: break
    # 35是爲了跳過image中的g
    index = line.find('G', 35)
    if (index == -1):
        index = line.find('g', 35)

    image_name = line[0: (index+1)]
    image = Image.open(image_name)
    # 图像数据保存在TFRecord文件中
    image_raw = image.tobytes()
    # 图像高度保存在TFRecord文件中
    # height = image.height
    # 图像宽度保存在TFRecord文件中
    # width = image.width
    # 对应图像数据的标签保存在TFRecord文件中
    label = line[index+6]  # Type str

    label = int(label)  # Type int 将label的数据类型从字符型转换为int型保存

    example = tf.train.Example(features = tf.train.Features(feature = {
        'image_raw': _bytes_feature(image_raw),
        'label': _int64_feature(label)
    }))
    writer.write(example.SerializeToString())
    image.close()

writer.close()
print('My TFRecords File is Finished!!!')
