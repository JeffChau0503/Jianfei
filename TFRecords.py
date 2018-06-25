# -*- coding: utf-8 -*-
import cv2
import os
import numpy as np
import tensorflow as tf


# 取回所有档案名称
def get_file(file_dir):
    # The images in each subfolder
    images = []
    # The subfolders
    subfolders = []

    # Using "os.walk" function to grab all the files in each folder
    for dirPath, dirNames, fileNames in os.walk(file_dir):
        for name in fileNames:
            images.append(os.path.join(dirPath, name))

        for name in dirNames:
            subfolders.append(os.path.join(dirPath, name))

    # To record the labels of the image dataset
    labels = []
    count = 0
    for a_folder in subfolders:
        n_img = len(os.listdir(a_folder))
        labels = np.append(labels, n_img * [count])
        count += 1

    subfolders = np.array([images, labels])
    subfolders = subfolders.transpose()

    image_list = list(subfolders[:, 0])
    label_list = list(subfolders[:, 1])
    label_list = [int(float(i)) for i in label_list]
    return image_list, label_list


# 轉Int64資料為 tf.train.Feature 格式
def int64_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


# 轉Bytes資料為 tf.train.Feature 格式
def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def convert_to_tfrecord(images, labels, filename):
    n_samples = len(labels)
    tfwriter = tf.python_io.TFRecordWriter(filename)

    print('\nTransform start...')
    for i in np.arange(0, n_samples):
        try:
            image = cv2.imread(images[i], 0)

            if image is None:
                print('Error image:' + images[i])
            else:
                image_raw = image.tostring()

            label = int(labels[i])

            # 將 tf.train.Feature 合併成 tf.train.Features
            ftrs = tf.train.Features(
                feature={'Label': int64_feature(label),
                         'image_raw': bytes_feature(image_raw)}
            )

            # 將 tf.train.Features 轉成 tf.train.Example
            example = tf.train.Example(features=ftrs)

            # 將 tf.train.Example 寫成 tfRecord 格式
            tfwriter.write(example.SerializeToString())
        except IOError as e:
            print('Skip!\n')

    tfwriter.close()
    print('Transform done!')


def main():
    # 資料集的位置
    train_dataset_dir = 'E:/imageset'

    # 取回所有檔案路徑
    images, labels = get_file(train_dataset_dir)

    # 開始寫入 TRRecord 檔案
    convert_to_tfrecord(images, labels, 'E:/imageset/Train.tfrecords')


if __name__ == '__main__':
    main()
