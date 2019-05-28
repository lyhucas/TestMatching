# -*- coding: utf-8 -*-

import os
import cv2
import time
import numpy as np
import tensorflow as tf

image_size = (160, 160)  # the size of resized images
image_channels = 1

out_feature_length = 320  # the length of feature

inputs = tf.placeholder(tf.float32, [None, image_size[0], image_size[1], image_channels])
keep_prob = tf.placeholder(tf.float32)
is_training = tf.placeholder(tf.bool)


def conv_layer(prev_layer, num_units, kernal_size, strides_size, is_training, name):
    conv_layer = tf.layers.conv2d(prev_layer, num_units, kernal_size, strides_size, padding='same',
                                  use_bias=True, activation=None)
    conv_layer = tf.layers.batch_normalization(conv_layer, training=is_training)
    conv_layer = tf.nn.elu(conv_layer, name=name)
    return conv_layer


def max_pool_2x2(prev_layer, name):
    return tf.layers.max_pooling2d(prev_layer, pool_size=[2, 2], strides=2, padding='same', name=name)


def inference_test(images, phase, keep_prob):
    # conv1
    with tf.variable_scope('conv1') as scope:
        conv1_1 = conv_layer(images, 20, 4, 1, phase, name=scope.name)
        conv1_2 = conv_layer(conv1_1, 40, 4, 1, phase, name=scope.name)

    # pool1
    with tf.variable_scope('pooling1') as scope:
        pool1 = max_pool_2x2(conv1_2, name=scope.name)

    # conv2
    with tf.variable_scope('conv2') as scope:
        conv2_1 = conv_layer(pool1, 40, 3, 1, phase, name=scope.name)
        conv2_2 = conv_layer(conv2_1, 80, 3, 1, phase, name=scope.name)

    # pool2
    with tf.variable_scope('pooling2') as scope:
        pool2 = max_pool_2x2(conv2_2, name=scope.name)

    # conv3
    with tf.variable_scope('conv3') as scope:
        conv3_1 = conv_layer(pool2, 80, 3, 1, phase, name=scope.name)
        conv3_2 = conv_layer(conv3_1, 120, 3, 1, phase, name=scope.name)

    # pool3
    with tf.variable_scope('pooling3') as scope:
        pool3 = max_pool_2x2(conv3_2, name=scope.name)

    # conv4
    with tf.variable_scope('conv4') as scope:
        conv4_1 = conv_layer(pool3, 120, 2, 2, phase, name=scope.name)
        conv4_2 = conv_layer(conv4_1, 160, 2, 1, phase, name=scope.name)

    # conv5
    with tf.variable_scope('conv5') as scope:
        conv5_1 = conv_layer(conv4_2, 160, 2, 2, phase, name=scope.name)
        conv5_2 = conv_layer(conv5_1, 320, 2, 1, phase, name=scope.name)


    # local3
    with tf.variable_scope('fc_1') as scope:
        reshape = tf.reshape(conv5_2, shape=[-1, 5 * 5 * 320])
        weights = tf.get_variable('weights',
                                  shape=[5 * 5 * 320, out_feature_length],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.004, dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[out_feature_length, ],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))

        fc_1 = tf.matmul(reshape, weights) + biases
        fc_1 = tf.layers.batch_normalization(fc_1, training=phase)

        local3 = tf.nn.elu(fc_1, name=scope.name)

    # dropout
    with tf.variable_scope('dropout') as scope:
        dropout_layer = tf.nn.dropout(local3, keep_prob, name=scope.name)

    return dropout_layer


def get_class_path(root_path, test_file):
    relative_image_path = []
    with open(test_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            dirname = line.strip()
            absolute_dirname = os.path.join(root_path, dirname)
            if not os.path.isdir(absolute_dirname):
                raise IOError('{} is not a directory!'.format(absolute_dirname))
            for img_name in os.listdir(absolute_dirname):
                relative_path = os.path.join(dirname, img_name)
                relative_image_path.append(relative_path)
    return relative_image_path  # 返回所有图像的相对路径，即类名/图像名


def read_image(root_path, test_file, resize=None):
    relative_image_path = get_class_path(root_path, test_file)
    width = image_size[0]
    height = image_size[1]
    temp = np.zeros([len(relative_image_path), width, height, 1], np.float32)

    for i, img_path in enumerate(relative_image_path):
        im = cv2.imread(os.path.join(root_path, img_path), cv2.IMREAD_GRAYSCALE)
        im = cv2.resize(im, (width, height))
        temp[i, :, :, 0] = im
    temp = np.asarray(temp, np.float32)
    return list(temp), relative_image_path


def extract_feature(session, layer, _LATEST_CHECKPOINT, root_path, test_file, once_read_imagenum=1):
    ims, relative_image_path = read_image(root_path, test_file)

    image_num = len(relative_image_path)
    total_iters = int(np.ceil(image_num / once_read_imagenum)) 

    all_feature_matrix = np.zeros((image_num, out_feature_length), dtype=np.float32)  

    print("begin to extract feature")
    saver = tf.train.Saver()
    saver.restore(session, _LATEST_CHECKPOINT)
    for iter in range(total_iters):
        start = iter * once_read_imagenum
        end = min(start + once_read_imagenum, image_num)
        batch_ims = ims[start:end]

        batch_image_features = session.run(layer, feed_dict={inputs: np.array(batch_ims), keep_prob: 1.0,
                                                             is_training: False})
        all_feature_matrix[start:end] = batch_image_features
        print('process {} images'.format(end))

    print("extract features done")
    return all_feature_matrix, relative_image_path


def get_feature_file(session, layer, _LATEST_CHECKPOINT, root_path, test_file, out_dir):
    all_feature_matrix, relative_image_path = extract_feature(session, layer, _LATEST_CHECKPOINT, root_path, test_file)
    image_num = all_feature_matrix.shape[0]
    assert len(relative_image_path) == image_num

    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n')
    print('match result:\n')
    for i in range(image_num):
        class_name = os.path.dirname(relative_image_path[i])
        save_dir = os.path.join(out_dir, class_name)
        if not os.path.isdir(save_dir):
            print('Create directory: {}'.format(save_dir))
            os.makedirs(save_dir)
        name_prefix = os.path.splitext(os.path.basename(relative_image_path[i]))[0]
        file_name = os.path.join(save_dir, name_prefix + '.fea')
        all_feature_matrix[i].tofile(file_name)
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print('process {} images!'.format(image_num))


if __name__ == '__main__':
    image_root_dir = '../test/FVC2002/Db1'     # the path of images

    test_file = '../test/data/Db1.txt'         # the test file

    CHECKPOINT_PATH = '../result/mix_dataset'       # the path of trained model
    LATEST_CHECKPOINT = tf.train.latest_checkpoint(CHECKPOINT_PATH)

    layer = inference_test(inputs, is_training, keep_prob)

    out_dir = '../test/fea_files/FVC2002/Db1'      # the path for feature files

    with tf.Session() as sess:
        get_feature_file(sess, layer, LATEST_CHECKPOINT, image_root_dir, test_file, out_dir)


