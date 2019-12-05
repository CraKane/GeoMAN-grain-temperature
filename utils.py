"""
@project = Predict the value according the data_geoMAN
@file = Training
@author = 10374
@create_time = 2019/11/19 19:20
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

import tensorflow as tf
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.util import nest
from sklearn import preprocessing
import numpy as np
import pandas as pd
import pickle as pk
import xlrd as xld


def Linear(args,
           output_size,
           bias,
           bias_initializer=None,
           kernel_initializer=None):
    """Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.
    Args:
      args: a 2D Tensor or a list of 2D, batch x n, Tensors.
      output_size: int, second dimension of W[i].
      bias: boolean, whether to add a bias term or not.
      bias_initializer: starting value to initialize the bias
        (default is all zeros).
      kernel_initializer: starting value to initialize the weight.
    Returns:
      A 2D Tensor with shape [batch x output_size] equal to
      sum_i(args[i] * W[i]), where W[i]s are newly created matrices.
    Raises:
      ValueError: if some of the arguments has unspecified or wrong shape.
    """
    if args is None or (nest.is_sequence(args) and not args):
        raise ValueError("`args` must be specified")
    if not nest.is_sequence(args):
        args = [args]

    # Calculate the total size of arguments on dimension 1.
    total_arg_size = 0
    shapes = [a.get_shape() for a in args]
    for shape in shapes:
        if shape.ndims != 2:
            raise ValueError(
                "linear is expecting 2D arguments: %s" % shapes)
        if shape[1].value is None:
            raise ValueError("linear expects shape[1] to be provided for shape %s, "
                             "but saw %s" % (shape, shape[1]))
        else:
            total_arg_size += shape[1].value

    dtype = [a.dtype for a in args][0]

    # Now the computation.
    scope = vs.get_variable_scope()
    with vs.variable_scope(scope) as outer_scope:
        weights = vs.get_variable(
            "kernel", [total_arg_size, output_size],
            dtype=dtype,
            initializer=kernel_initializer)
        if len(args) == 1:
            res = math_ops.matmul(args[0], weights)
        else:
            res = math_ops.matmul(array_ops.concat(args, 1), weights)
        if not bias:
            return res
        with vs.variable_scope(outer_scope) as inner_scope:
            inner_scope.set_partitioner(None)
            if bias_initializer is None:
                bias_initializer = init_ops.constant_initializer(
                    0.0, dtype=dtype)
            biases = vs.get_variable(
                "bias", [output_size],
                dtype=dtype,
                initializer=bias_initializer)
        return nn_ops.bias_add(res, biases)

# initial the basic training hyperparameters
# the proposed number of hidden layer is 128
def basic_hyperparams():
    return tf.contrib.training.HParams(
        # GPU arguments
        gpu_id='0',

        # model parameters
        learning_rate=1e-3,
        lambda_l2_reg=1e-3,
        gc_rate=2.5,  # to avoid gradient exploding
        dropout_rate=0.3,
        n_stacked_layers=2,
        s_attn_flag=2,
        ext_flag=True,

        # encoder parameter
        n_sensors=174,  # find the global input  total: 200 points
        n_input_encoder=26,  # point 84, 85, 86, 88, 89, 90, 92, 93,
        # 94, 104, 105, 106, 108, 110, 112, 113, 114, 124, 125, 126,
        # 128, 129, 130, 132, 133, 134
        n_steps_encoder=12,  # how many historical time steps we use for predictions
        n_hidden_encoder=128,  # size of hidden units

        # decoder parameter
        n_input_decoder=1,
        n_external_input=8, # EVP, GST, PRE, PRS, RHU, SSD, TEM, WIN
        n_steps_decoder=6 ,  # how many future time steps we predict
        n_hidden_decoder=128,
        n_output_decoder=1  # size of the decoder output
    )


# calculate the total number of parameters
def count_total_params():
    """ count the parameters in the model """
    total_parameters = 0
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        # print(shape)
        # print(len(shape))
        variable_parameters = 1
        for dim in shape:
            # print(dim)
            variable_parameters *= dim.value
        # print(variable_parameters)
        total_parameters += variable_parameters
    print(total_parameters)


# the function for loading data
def load_data(input_path, mode, n_steps_encoder, n_steps_decoder):
    """ load training/validation data
    Args:
        input_path:
        mode: "train" or "test"
        n_steps_encoder: length of encoder, i.e., how many historical time steps we use for predictions
        n_steps_decoder: length of decoder, i.e., how many future time steps we predict
    Return:
        a list
    """
    mode_local_inp = np.load(  # load the file with data: the local input, shape is (?, 12, 26)
        input_path + "GeoMAN-{}-{}-{}-local_inputs.npy".format(n_steps_encoder, n_steps_decoder, mode))
    global_attn_index = np.load(  # load the file with data: the global inputs index,
                                  # for selecting the global attention states, shape is (?, 1)
        input_path + "GeoMAN-{}-{}-{}-global_attn_state_indics.npy".format(n_steps_encoder, n_steps_decoder, mode))
    global_inp_index = np.load(  # load the file with data: the global inputs index,
                                 # for selecting the global input, shape is (?, 1)
        input_path + "GeoMAN-{}-{}-{}-global_input_indics.npy".format(n_steps_encoder, n_steps_decoder, mode))
    mode_ext_inp = np.load(  # load the file with data: the externel input, shape is (?, 6, 8)
        input_path + "GeoMAN-{}-{}-{}-external_inputs.npy".format(n_steps_encoder, n_steps_decoder, mode))
    mode_labels = np.load(  # load the file with data: the predict label, shape is (?, 6, 1)
        input_path + "GeoMAN-{}-{}-{}-decoder_gts.npy".format(n_steps_encoder, n_steps_decoder, mode))
    return [mode_local_inp, global_inp_index, global_attn_index, mode_ext_inp, mode_labels]


# the function for shuffling data 打乱数据
# reduce the correlation of data
def shuffle_data(training_data):
    """ shuffle data"""
    shuffle_index = np.random.permutation(training_data[0].shape[0])
    new_training_data = []
    for inp in training_data:
        new_training_data.append(inp[shuffle_index])
    return new_training_data


# construct the training feed data dictionary
def get_batch_feed_dict(model, k, batch_size, training_data, global_inputs, global_attn_states):
    """ get feed_dict of each batch in a training epoch"""
    train_local_inp = training_data[0]
    train_global_inp = training_data[1]
    train_global_attn_ind = training_data[2]
    train_ext_inp = training_data[3]
    train_labels = training_data[4]
    n_steps_encoder = train_local_inp.shape[1]

    batch_local_inp = train_local_inp[k:k + batch_size]
    batch_ext_inp = train_ext_inp[k:k + batch_size]
    batch_labels = train_labels[k:k + batch_size]
    batch_labels = np.expand_dims(batch_labels, axis=2)
    batch_global_inp = train_global_inp[k:k + batch_size]
    batch_global_attn = train_global_attn_ind[k:k + batch_size]
    tmp = []
    for j in batch_global_inp:
        tmp.append(
            global_inputs[j: j + n_steps_encoder, :])
    tmp = np.array(tmp)
    feed_dict = {model.phs['local_inputs']: batch_local_inp,
                 model.phs['global_inputs']: tmp,
                 model.phs['local_attn_states']: np.swapaxes(batch_local_inp, 1, 2),
                 model.phs['global_attn_states']: global_attn_states[batch_global_attn],
                 model.phs['external_inputs']: batch_ext_inp,
                 model.phs['labels']: batch_labels}
    return feed_dict


# load the global inputs & attention states
def load_global_inputs(input_path, n_steps_encoder, n_steps_decoder):
    """ load global inputs"""
    global_inputs = np.load(  # load the file with data: the global input, shape is (?, 174)
        input_path + "GeoMAN-{}-{}-global_inputs.npy".format(n_steps_encoder, n_steps_decoder))
    global_attn_states = np.load(  # load the file with data: the global attention states, shape is (?, 174, 26, 12)
        input_path + "GeoMAN-{}-{}-global_attn_state.npy".format(n_steps_encoder, n_steps_decoder))
    return global_inputs, global_attn_states


# construct the validation feed data dictionary
def get_valid_batch_feed_dict(model, valid_indexes, k, valid_data, global_inputs, global_attn_states):
    """ get feed_dict of each batch in the validation set"""
    valid_local_inp = valid_data[0]
    valid_global_inp = valid_data[1]
    valid_global_attn_ind = valid_data[2]
    valid_ext_inp = valid_data[3]
    valid_labels = valid_data[4]
    n_steps_encoder = valid_local_inp.shape[1]

    batch_local_inp = valid_local_inp[valid_indexes[k]:valid_indexes[k + 1]]
    batch_ext_inp = valid_ext_inp[valid_indexes[k]:valid_indexes[k + 1]]
    batch_labels = valid_labels[valid_indexes[k]:valid_indexes[k + 1]]
    batch_labels = np.expand_dims(batch_labels, axis=2)
    batch_global_inp = valid_global_inp[valid_indexes[k]:valid_indexes[k + 1]]
    batch_global_attn = valid_global_attn_ind[valid_indexes[k]:valid_indexes[k + 1]]
    tmp = []
    for j in batch_global_inp:
        tmp.append(
            global_inputs[j: j + n_steps_encoder, :])
    tmp = np.array(tmp)
    feed_dict = {model.phs['local_inputs']: batch_local_inp,
                 model.phs['global_inputs']: tmp,
                 model.phs['local_attn_states']: np.swapaxes(batch_local_inp, 1, 2),
                 model.phs['global_attn_states']: global_attn_states[batch_global_attn],
                 model.phs['external_inputs']: batch_ext_inp,
                 model.phs['labels']: batch_labels}
    return feed_dict


# split the data to train, validation, test and pred
def handle_data(local_inputs, externel_inputs, label_inputs, global_inputs,
                global_attn_inputs, n_steps_encoder=12, n_steps_decoder=6):
    input_path = './'
    # the number of training data is 892
    # the number of validation data is 233
    # the number of test data is 233
    # means 4:1:1
    num_train = 892; num_test = 223; num_valid = 223

    # splited train local inputs
    global_input_indics = np.int64(  # the first 892 data
        np.linspace(0, num_train-1, num=num_train))
    np.save(
        input_path + "GeoMAN-{}-{}-{}-local_inputs.npy".format(n_steps_encoder, n_steps_decoder, 'train'), local_inputs[:num_train])
    np.save(
        input_path + "GeoMAN-{}-{}-{}-global_attn_state_indics.npy".format(n_steps_encoder, n_steps_decoder, 'train'), global_input_indics)
    np.save(
        input_path + "GeoMAN-{}-{}-{}-global_input_indics.npy".format(n_steps_encoder, n_steps_decoder, 'train'), global_input_indics)
    np.save(
        input_path + "GeoMAN-{}-{}-{}-external_inputs.npy".format(n_steps_encoder, n_steps_decoder, 'train'), externel_inputs[:num_train])
    np.save(
        input_path + "GeoMAN-{}-{}-{}-decoder_gts.npy".format(n_steps_encoder, n_steps_decoder, 'train'), label_inputs[:num_train])

    # splited validation local inputs
    global_input_valid_indics = np.int64(  # the after 223 data
        np.linspace(892, 892+num_valid-1, num=num_valid))
    np.save(
        input_path + "GeoMAN-{}-{}-{}-local_inputs.npy".format(n_steps_encoder, n_steps_decoder, 'eval'),
        local_inputs[num_train:num_train+num_valid])
    np.save(
        input_path + "GeoMAN-{}-{}-{}-global_attn_state_indics.npy".format(n_steps_encoder, n_steps_decoder, 'eval'),
        global_input_valid_indics)
    np.save(
        input_path + "GeoMAN-{}-{}-{}-global_input_indics.npy".format(n_steps_encoder, n_steps_decoder, 'eval'),
        global_input_valid_indics)
    np.save(
        input_path + "GeoMAN-{}-{}-{}-external_inputs.npy".format(n_steps_encoder, n_steps_decoder, 'eval'),
        externel_inputs[num_train:num_train+num_valid])
    np.save(
        input_path + "GeoMAN-{}-{}-{}-decoder_gts.npy".format(n_steps_encoder, n_steps_decoder, 'eval'),
        label_inputs[num_train:num_train+num_valid])

    # splited test local inputs
    global_input_test_indics = np.int64(  # the last 223 data
        np.linspace(1115, 1115+num_test-1, num=num_test))
    np.save(
        input_path + "GeoMAN-{}-{}-{}-local_inputs.npy".format(n_steps_encoder, n_steps_decoder, 'test'),
        local_inputs[num_train+num_valid:])
    np.save(
        input_path + "GeoMAN-{}-{}-{}-global_attn_state_indics.npy".format(n_steps_encoder, n_steps_decoder, 'test'),
        global_input_test_indics)
    np.save(
        input_path + "GeoMAN-{}-{}-{}-global_input_indics.npy".format(n_steps_encoder, n_steps_decoder, 'test'),
        global_input_test_indics)
    np.save(
        input_path + "GeoMAN-{}-{}-{}-external_inputs.npy".format(n_steps_encoder, n_steps_decoder, 'test'),
        externel_inputs[num_train+num_valid:])
    np.save(
        input_path + "GeoMAN-{}-{}-{}-decoder_gts.npy".format(n_steps_encoder, n_steps_decoder, 'test'),
        label_inputs[num_train+num_valid:])

    # global inputs
    np.save(
        input_path + "GeoMAN-{}-{}-global_inputs.npy".format(n_steps_encoder, n_steps_decoder), global_inputs)
    np.save(
        input_path + "GeoMAN-{}-{}-global_attn_state.npy".format(n_steps_encoder, n_steps_decoder), global_attn_inputs)


# save the label normalization as a scaler to the scalers folder
def split_labels(label_):
    # preprocess the labels
    X = label_

    # create a standard scaler
    scaler = preprocessing.StandardScaler().fit(X)

    # save the standard scaler as .pkl file
    with open('./scalers/scaler-0.pkl', 'wb') as f:
        # save the scaler
        pk.dump(scaler, f)

        # close the file
        f.close()

    print("Save the scaler successfully!")