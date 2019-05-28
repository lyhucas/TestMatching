#-*- coding: utf-8 -*-
import time
import os
import numpy as np

from sklearn.svm import SVC
from sklearn.externals import joblib

import tensorflow as tf
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

out_feature_length = 320
BATCH_SIZE = 1          

DEBUG = False                     


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


def inference_deploy_nn(nn_images, keep_prob):
    with tf.variable_scope('dense1') as scope:
        dense1 = tf.layers.dense(inputs=nn_images, units=640, activation=tf.nn.relu, name=scope.name)
    
    with tf.variable_scope('nn_dropout') as scope:
        dropout_layer = tf.nn.dropout(dense1, keep_prob, name=scope.name)
    
    with tf.variable_scope('dense2') as scope:
        logits = tf.layers.dense(inputs=dropout_layer, units=2, activation=None, name=scope.name)
    
    return logits
    
def _split_pairs(data_path, pair_file):
    datapath1 = []  
    datapath2 = []  
    label_lists = []  
    with open(pair_file, 'r') as pair_txt:
        lines = pair_txt.readlines()
        for line in lines:
            line = line.strip().split()
            if len(line) == 3: 
                dir, file1, file2 = line
                if dir[0] != '/':  
                    dir = os.path.join(data_path, dir)
                filepath1 = os.path.join(dir, file1)   
                filepath2 = os.path.join(dir, file2)
                datapath1.append(filepath1)
                datapath2.append(filepath2)
                label = float(1)
                label_lists.append(label) 
            elif len(line) == 4:  
                dir1, file1, dir2, file2 = line
                if dir1[0] != '/':
                    dir1 = os.path.join(data_path, dir1)
                if dir2[0] != '/':
                    dir2 = os.path.join(data_path, dir2)
                filepath1 = os.path.join(dir1, file1)
                filepath2 = os.path.join(dir2, file2)
                datapath1.append(filepath1)
                datapath2.append(filepath2)
                label = float(0)
                label_lists.append(label)  
            if DEBUG:
                print("left image:{:s}\tright image:{:s}\tlabel:{}\n".format(filepath1, filepath2, label))
    
    return datapath1, datapath2, label_lists


def get_fea_file(data_path, pair_file, _feature_dir):
    all_left_image_datapath, all_right_image_datapaths, label_lists = _split_pairs(
        data_path, pair_file) 
    all_images_nums = len(all_left_image_datapath)  
    assert (len(all_left_image_datapath) != 0 and len(
        all_left_image_datapath) == len(all_right_image_datapaths))
    
    features_matrix = np.zeros((all_images_nums, out_feature_length*2), dtype=np.float32)
    
    name2feature = {}
    for _line in os.listdir(_feature_dir):
        fea_path = os.path.join(_feature_dir, _line)
        for _file in os.listdir(fea_path):
            file_path = os.path.join(fea_path, _file)
            feature = np.fromfile(file_path, dtype=np.float32)
            temp_name = os.path.splitext(_file)[0]
            name2feature[temp_name] = feature    

    for i in range(len(all_left_image_datapath)):
        _left_file = os.path.basename(all_left_image_datapath[i])
        left_file_name = os.path.splitext(_left_file)[0]
        if left_file_name not in name2feature.keys():
            raise ValueError('{} is not in test_files!'.format(left_file_name))
        left_images_features = name2feature[left_file_name]
        left_images_features = np.reshape(left_images_features, (1, out_feature_length)
                                          ) 
        
        _right_file = os.path.basename(all_right_image_datapaths[i])
        right_file_name = os.path.splitext(_right_file)[0]
        if right_file_name not in name2feature.keys():
            raise ValueError('{} is not in test_files!'.format(right_file_name))
        right_images_features = name2feature[right_file_name]
        right_images_features = np.reshape(right_images_features, (1, out_feature_length)
                                          )  

        _feature = np.concatenate(
            (left_images_features, right_images_features)) 
        new_feature = np.reshape(_feature, (-1, out_feature_length*2))

        features_matrix[i, :] = new_feature  
    label_matrix = np.asarray(label_lists, np.float32)
    return features_matrix, label_matrix, all_left_image_datapath, all_right_image_datapaths


def predict(nn_images, _LATEST_CHECKPOINT, test_data_path, test_pair_file, _feature_dir):
    features_matrix, label_matrix, all_test_left_image_datapaths, all_test_right_image_datapaths = get_fea_file(
        test_data_path, test_pair_file, _feature_dir)

    NN_layer, regularizers = inference_test_nn(nn_images)
    nn_saver = tf.train.Saver()

    session = tf.Session()
    nn_saver.restore(session, _LATEST_CHECKPOINT)
    
    out = session.run(NN_layer, {nn_images: features_matrix})
    predicts = tf.nn.softmax(tf.cast(out, tf.float32))

    predict_label_matrix = np.float32(session.run(predicts))
    
    session.close()

    test_data_num = features_matrix.shape[0]
    assert predict_label_matrix.shape[0] == test_data_num
    true_to_pos = 0  
    true_to_neg = 0   
    false_to_pos = 0 
    false_to_neg = 0 
    for i in range(0, test_data_num):

        if predict_label_matrix[i][1] > 0.5:
            predict = 1
        else:
            predict = 0
        label = label_matrix[i]
        if label == 1:  
            if predict == 1:  
                true_to_pos += 1
            else:
                true_to_neg += 1
        else:
            if predict == 1:
                false_to_pos += 1
            else:
                false_to_neg += 1
    print("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("total: {} pair pictures,{} pair match,{} pair unmatch".format(test_data_num, (true_to_pos + true_to_neg),
                                                                   (false_to_pos + false_to_neg)))
    print("accuracy: {}".format(1.0 * (true_to_pos + false_to_neg) / test_data_num))
    print("true --> true: {}".format(1.0 * true_to_pos / (true_to_pos + true_to_neg)))
    print("false --> false: {}".format(1.0 * false_to_neg / (false_to_pos + false_to_neg)))
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    return predict_label_matrix, label_matrix, all_test_left_image_datapaths, all_test_right_image_datapaths

if __name__ == '__main__':
    T1 = time.clock()

    image_root_dir = '../test/FVC2002/Db1'     # the path of images

    test_pair_file = '../test/data/test_pairs.txt'       # pair data file for time testing

    feature_dir = '../test/fea_files/FVC2002/Db1'       # the path of feature files

    NN_CHECKPOINT_PATH = '../NN_result'         # the path of trained classifier model
    NN_LATEST_CHECKPOINT = tf.train.latest_checkpoint(NN_CHECKPOINT_PATH)

    nn_input = tf.placeholder(tf.float32, [None, out_feature_length*2])
    predict(nn_input, NN_LATEST_CHECKPOINT, image_root_dir, test_pair_file, feature_dir)
    print("spend {} seconds averagely".format(time.clock() - T1))

