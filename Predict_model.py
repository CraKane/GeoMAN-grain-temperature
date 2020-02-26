"""
@project = Predict the value according the data_geoMAN
@file = Predict
@author = 10374
@create_time = 2019/11/19 19:19
@inproceedings{ijcai2018-476,
  title     = {GeoMAN: Multi-level Attention Networks for Geo-sensory Time Series Prediction},
  author    = {Yuxuan Liang and Songyu Ke and Junbo Zhang and Xiuwen Yi and Yu Zheng},
  booktitle = {Proceedings of the Twenty-Seventh International Joint Conference on
               Artificial Intelligence, {IJCAI-18}},
  publisher = {International Joint Conferences on Artificial Intelligence Organization},
  pages     = {3428--3434},
  year      = {2018},
  month     = {7},
  doi       = {10.24963/ijcai.2018/476},
  url       = {https://doi.org/10.24963/ijcai.2018/476},
}
"""

import os
import tensorflow as tf
import numpy as np
import pandas as pd
import keras as ks
import scipy as sp
import pickle
from utils import basic_hyperparams
from utils import load_data
from utils import load_global_inputs
from utils import get_valid_batch_feed_dict
from N_network import GeoMAN
import xlrd as xld
from sklearn import preprocessing
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

if __name__ == '__main__':
    # use specific gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True

    # load hyperparameters
    session = tf.Session(config=tf_config)
    hps = basic_hyperparams()  # test parameters

    # model construction
    tf.reset_default_graph()
    print(hps)
    model = GeoMAN(hps)

    # read data from test set
    input_path = './data/'
    test_data = load_data(
        input_path, 'test', hps.n_steps_encoder, hps.n_steps_decoder)
    global_inputs, global_attn_states = load_global_inputs(
        input_path, hps.n_steps_encoder, hps.n_steps_decoder)
    num_test = len(test_data[0])
    print('test samples: {0}'.format(num_test))

    # read scaler of the labels
    f = open('./data/scalers/scaler-0.pkl', 'rb')
    scaler = pickle.load(f)
    f.close()

    # path
    if hps.ext_flag:
        if hps.s_attn_flag == 0:
            model_name = 'GeoMANng'
        elif hps.s_attn_flag == 1:
            model_name = 'GeoMANnl'
        else:
            model_name = 'GeoMAN'
    else:
        model_name = 'GeoMANne'
    model_path = './logs/{}-{}-{}-{}-{}-{:.2f}-{:.3f}/'.format(model_name,
                                                               hps.n_steps_encoder,
                                                               hps.n_steps_decoder,
                                                               hps.n_stacked_layers,
                                                               hps.n_hidden_encoder,
                                                               hps.dropout_rate,
                                                               hps.lambda_l2_reg)
    model_path += 'saved_models/final_model.ckpt'


    # restore model
    print("Starting loading model...")
    saver = tf.train.Saver()
    with tf.Session() as sess:
        model.init(sess)

        # Restore model weights from previously saved model
        saver.restore(sess, model_path)
        print("Model successfully restored from file: %s" % model_path)

        # open the xlsx workbook
        workbook_grain = xld.open_workbook("./data/graintem_1.xlsx")
        worksheet_grain = workbook_grain.sheet_by_name("温度值")
        # read externel data
        workbook = xld.open_workbook("./data/mete.xls")
        worksheet = workbook.sheet_by_name("Sheet1")

        # test
        local_inputs = []; global_inputs = []; global_attn_states = []
        externel_inputs = []; labels_inputs = [np.zeros((6, 1))]
        # construct the data
        # read the local input
        local_index = [85, 86, 87, 89, 90, 91, 93, 94, 95, 105, 106,
                       107, 109, 111, 113, 114, 115, 125, 126, 127, 129,
                       130, 131, 133, 134, 135]
        local_input = [];
        for i in range(len(local_index)):
            col_data = worksheet_grain.col_values(local_index[i])  # the i-th column
            col_data = np.array(col_data[1339:1351]).reshape(-1, 1)
            scaler = preprocessing.StandardScaler().fit(col_data)
            col_data = np.array(scaler.transform(col_data)).reshape(1, -1)
            local_input.insert(i, col_data[0])
        local_input = np.array(local_input)
        local_input = local_input.T
        local_inputs.append(local_input)
        local_inputs = np.array(local_inputs)

        # read the externel input
        externel_input = [];
        nclos = worksheet.ncols
        for i in range(0, nclos - 1):
            col_data = worksheet.col_values(i + 1)  # the i-th column
            col_data = [float(x) for x in col_data[1442:1448]]
            col_data = np.array(col_data).reshape(-1, 1)
            scaler = preprocessing.StandardScaler().fit(col_data)
            col_data = np.array(scaler.transform(col_data)).reshape(1, -1)[0]
            externel_input.insert(i, col_data)
        externel_input = np.array(externel_input)
        externel_input = externel_input.T
        externel_inputs.append(externel_input)
        externel_inputs = np.array(externel_inputs)

        # read the global data
        global_input = []; global_attn_input = [];
        global_index = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
                        18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
                        34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49,
                        50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65,
                        66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81,
                        82, 83, 84, 88, 92, 96, 97, 98, 99, 100, 101, 102, 103, 104, 108,
                        110, 112, 116, 117, 118, 119, 120, 121, 122, 123, 124, 128, 132,
                        136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148,
                        149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161,
                        162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174,
                        175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187,
                        188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199]

        for i in range(len(global_index)):
            col_data = worksheet_grain.col_values(global_index[i] + 1)  # the i-th column
            col_data = np.array(col_data[1339:1351]).reshape(-1, 1)
            scaler = preprocessing.StandardScaler().fit(col_data)
            col_data = np.array(scaler.transform(col_data)).reshape(1, -1)
            global_input.insert(i, col_data[0])
        global_input = np.array(global_input)
        for i in range(148):
            global_attn_input.insert(i, global_input[i:i + 26])
        for i in range(148, 174):
            global_attn_input.insert(i, global_input[148:174])
        global_input = global_input.T
        global_inputs.append(global_input)
        global_inputs = np.array(global_inputs)
        global_attn_input = np.array(global_attn_input)
        global_attn_states.append(global_attn_input)
        global_attn_states = np.array(global_attn_states)

        # construct the feed dictionary
        feed_dict = {model.phs['local_inputs']: local_inputs,
                     model.phs['global_inputs']: global_inputs,
                     model.phs['local_attn_states']: np.swapaxes(local_inputs, 1, 2),
                     model.phs['global_attn_states']: global_attn_states,
                     model.phs['external_inputs']: externel_inputs,
                     model.phs['labels']: labels_inputs}
        # re-scale predicted labels
        batch_preds = sess.run(model.phs['preds'], feed_dict)
        batch_preds = np.swapaxes(batch_preds, 0, 1)
        batch_preds = np.reshape(batch_preds, [batch_preds.shape[0], -1])
        batch_preds = scaler.inverse_transform(batch_preds)-9
        preds = batch_preds

    print('===============PREDICT===============')
    print(preds)


