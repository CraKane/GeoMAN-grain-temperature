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
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# the function of calculating root_mean_squared_error
def root_mean_squared_error(labels, preds):
    total_size = np.size(labels)
    return np.sqrt(np.sum(np.square(labels - preds)) / total_size)


# the function of calculating mean_absolute_error
def mean_absolute_error(labels, preds):
    total_size = np.size(labels)
    return np.sum(np.abs(labels - preds)) / total_size


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

    # test params
    test_rmses = []
    test_maes = []

    # restore model
    print("Starting loading model...")
    saver = tf.train.Saver()
    with tf.Session() as sess:
        model.init(sess)

        # Restore model weights from previously saved model
        saver.restore(sess, model_path)
        print("Model successfully restored from file: %s" % model_path)

        # test
        # display the test loss
        test_loss = 0
        test_indexes = np.int64(
            np.linspace(0, num_test - 1, num=num_test))
        for k in range(num_test - 1):
            feed_dict = get_valid_batch_feed_dict(
                model, test_indexes, k, test_data, global_inputs, global_attn_states)
            # re-scale predicted labels
            batch_preds = sess.run(model.phs['preds'], feed_dict)
            batch_preds = np.swapaxes(batch_preds, 0, 1)
            batch_preds = np.reshape(batch_preds, [batch_preds.shape[0], -1])
            batch_preds = scaler.inverse_transform(batch_preds)
            # re-scale real labels
            batch_labels = test_data[4]
            batch_labels = batch_labels[test_indexes[k]:test_indexes[k + 1]]
            batch_labels = scaler.inverse_transform(batch_labels)
            # calculate the root_mean_squared_error
            test_rmses.append(root_mean_squared_error(
                batch_labels, batch_preds))
            # calculate the mean_absolute_error
            test_maes.append(mean_absolute_error(batch_labels, batch_preds))

    test_rmses = np.asarray(test_rmses)
    test_maes = np.asarray(test_maes)

    print('===============METRIC===============')
    print('rmse = {:.6f}'.format(test_rmses.mean()))
    print('mae = {:.6f}'.format(test_maes.mean()))
