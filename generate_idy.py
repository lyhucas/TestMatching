#-*- coding: utf-8 -*-

import os
import cv2
import time
import numpy as np 
import tensorflow as tf



out_feature_length = 320

def inference_test_nn(nn_images):
    with tf.variable_scope('dense1') as scope:
        weights = tf.get_variable('weights',
                                  shape=[out_feature_length*2, out_feature_length*2],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.01, dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[out_feature_length*2, ],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))

        fc_1 = tf.matmul(nn_images, weights) + biases
        local3 = tf.nn.relu(fc_1, name=scope.name)
        regularizer_1 = tf.nn.l2_loss(weights) + tf.nn.l2_loss(biases)

    with tf.variable_scope('dense2') as scope:
        weights = tf.get_variable('weights',
                                  shape=[out_feature_length*2, 2],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.01, dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[2, ],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))

        logits = tf.matmul(local3, weights) + biases
        regularizer_2 = tf.nn.l2_loss(weights) + tf.nn.l2_loss(biases)

    regularizers = (regularizer_1 + regularizer_2)

    return logits, regularizers

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
    return relative_image_path

def gene_feature_dict(_fea_root_path):
    _name2feature = {}
    fea_class_files = os.listdir(_fea_root_path)
    for line in fea_class_files:
        class_path = os.path.join(_fea_root_path, line)
        for fea_file in os.listdir(class_path):
            file_path = os.path.join(class_path, fea_file)
            name_prefix = os.path.splitext(fea_file)[0]
            feature = np.fromfile(file_path, dtype=np.float32)
            _name2feature[name_prefix] = feature
            
    return _name2feature

def get_idy_file(root_path, _fea_root_path, nn_input, _LATEST_CHECKPOINT, test_file, out_dir):
    relative_image_path = get_class_path(root_path, test_file)
    name2feature = gene_feature_dict(_fea_root_path)
    
    filename_list = []
    for line in relative_image_path:
        _name_prefix = os.path.splitext(os.path.basename(line))[0]
        filename_list.append(_name_prefix)
    
    image_num = len(relative_image_path)
    assert len(filename_list) == image_num

    if not os.path.isdir(out_dir):
        print('Create directory: {}'.format(out_dir))
        os.makedirs(out_dir)

    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n')

    temp_files = os.listdir(out_dir)
    NN_layer, regularizers = inference_test_nn(nn_input)
    nn_saver = tf.train.Saver()

    session = tf.Session()
    nn_saver.restore(session, _LATEST_CHECKPOINT)

    for i in range(image_num):
        name_prefix = filename_list[i]
        temp_name = name_prefix + '.txt'
        if temp_name not in temp_files:
            input_feature_matrix = [np.concatenate((name2feature[filename_list[i]], name2feature[filename_list[j]])
                ) for j in range(image_num)]
            input_feature_matrix = np.array(input_feature_matrix).reshape(
                (image_num, 2 * out_feature_length))

            out = session.run(NN_layer, {nn_input: input_feature_matrix})

            predicts = tf.nn.softmax(tf.cast(out, tf.float32))

            predict_matrix = np.float32(session.run(predicts))

            assert image_num == predict_matrix.shape[0]

            file_name = os.path.join(out_dir, temp_name)
            with open(file_name, 'w') as file: 
                for j in range(image_num):
                    if i != j:
                        match_image_name = filename_list[j]
                        if os.path.dirname(relative_image_path[i]) == os.path.dirname(relative_image_path[j]):
                            label = 1
                        else:
                            label = 0
                        out_line = "{:s}\t{:d}\t{:.5f}".format(match_image_name, label, predict_matrix[j][-1])
                        file.writelines(out_line + '\n')
                    else:
                        pass

            print('generate {} txt file'.format(i))
        else:
            print('file {0} exits'.format(temp_name))
    session.close()
    print('process {} images!'.format(image_num))

if __name__ == '__main__':
    image_root_dir = '../test/FVC2002/Db1'     # the path of images
    test_file = '../test/data/Db1.txt'         # the test file

    CHECKPOINT_PATH = '../NN_result'           # the path of trained classifier model
    LATEST_CHECKPOINT = tf.train.latest_checkpoint(CHECKPOINT_PATH)

    feature_input = tf.placeholder(tf.float32, [None, out_feature_length*2])

    feature_root_path = '../test/fea_files/FVC2002/Db1'        # the path of feature files

    save_idy_file_dir = '../test/test_files/FVC2002/Db1'       # the path for matching files

    get_idy_file(image_root_dir, feature_root_path, feature_input, LATEST_CHECKPOINT, test_file, save_idy_file_dir)


