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
import numpy as np
import pandas as pd
import os
import json
import pickle
from utils import load_data
from utils import load_global_inputs
from utils import basic_hyperparams
from N_network import GeoMAN
from utils import get_batch_feed_dict
from utils import shuffle_data
from utils import get_valid_batch_feed_dict


if __name__ == '__main__':
	np.random.seed(2017)

	# use specific gpu
	os.environ["CUDA_VISIBLE_DEVICES"] = "0"
	tf_config = tf.ConfigProto()
	tf_config.gpu_options.allow_growth = True
	session = tf.Session(config=tf_config)

	# load hyperparameters
	hps = basic_hyperparams()
	print(hps)

	# read data from different sets
	input_path = './data/'
	training_data = load_data(
		input_path, 'train', hps.n_steps_encoder, hps.n_steps_decoder)
	valid_data = load_data(  # validation data
		input_path, 'eval', hps.n_steps_encoder, hps.n_steps_decoder)
	global_inputs, global_attn_states = load_global_inputs(
		input_path, hps.n_steps_encoder, hps.n_steps_decoder)

	# print dataset info
	num_train = len(training_data[0])
	num_valid = len(valid_data[0])
	print('train samples: {0}'.format(num_train))
	print('eval samples: {0}'.format(num_valid))

	# model construction
	tf.reset_default_graph()
	model = GeoMAN(hps)

	# print trainable params
	for i in tf.trainable_variables():
		print(i)

	# print all placeholders
	phs = [x for x in tf.get_default_graph().get_operations()
		   if x.type == "Placeholder"]
	print(phs)

	# count the parameters in our model
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
	print('total parameters: {}'.format(total_parameters))

	# path for log saving
	if hps.ext_flag:
		if hps.s_attn_flag == 0:
			model_name = 'GeoMANng'
		elif hps.s_attn_flag == 1:
			model_name = 'GeoMANnl'
		else:
			model_name = 'GeoMAN'
	else:
		model_name = 'GeoMANne'
	logdir = './logs/{}-{}-{}-{}-{}-{:.2f}-{:.3f}/'.format(model_name,
														   hps.n_steps_encoder,
														   hps.n_steps_decoder,
														   hps.n_stacked_layers,
														   hps.n_hidden_encoder,
														   hps.dropout_rate,
														   hps.lambda_l2_reg)
	model_dir = logdir + 'saved_models/'
	if not os.path.exists(logdir):
		os.mkdir(logdir)
	if not os.path.exists(model_dir):
		os.mkdir(model_dir)
	results_dir = logdir + 'results/'

	# train params
	total_epoch = 1000
	batch_size = 40
	display_iter = 4# 500 How many times saving the model
	save_log_iter = 10# 100 How many times saving the log
	valid_losses = [np.inf]

	# training process
	with tf.Session() as sess:
		# contruct a tensorflow saver node
		saver = tf.train.Saver()

		# initialize
		model.init(sess)
		iter = 0
		summary_writer = tf.summary.FileWriter(logdir)

		# start training
		for i in range(total_epoch):
			print('----------epoch {}-----------'.format(i))
			training_data = shuffle_data(training_data)
			iiter = 0
			for j in range(i%892, num_train, batch_size):  # take the data in a batch: length is batch_size
				print('------data_iter {}-------'.format(iiter))
				iter += 1
				# here is training
				feed_dict = get_batch_feed_dict(
					model, j, batch_size, training_data, global_inputs, global_attn_states)

				_, merged_summary = sess.run(
					[model.phs['train_op'], model.phs['summary']], feed_dict)
				# summary_writer.add_summary(merged_summary, iter)
				if iter % save_log_iter == 0:
					summary_writer.add_summary(merged_summary, iter)

				# display the valid loss
				if iter % display_iter == 0:
					valid_loss = 0
					valid_indexes = np.int64(
						np.linspace(0, num_valid - 1, num=num_valid))
					# print(valid_indexes)
					for k in range(num_valid - 1):
						feed_dict = get_valid_batch_feed_dict(
							model, valid_indexes, k, valid_data, global_inputs, global_attn_states)

						batch_loss = sess.run(model.phs['loss'], feed_dict)
						# accumulate the loss
						valid_loss += batch_loss
					# get the mean
					valid_loss /= num_valid
					valid_losses.append(valid_loss)
					valid_loss_sum = tf.Summary(
						value=[tf.Summary.Value(tag="valid_loss", simple_value=valid_loss)])
					summary_writer.add_summary(valid_loss_sum, iter)

					# if the new loss is less than the old, update the model & update the loss
					if valid_loss < min(valid_losses[:-1]):
						print('iter {}\tvalid_loss = {:.6f}\tmodel saved!!'.format(
							iter, valid_loss))
						saver.save(sess, model_dir +
								   'model_{}.ckpt'.format(iter))
						saver.save(sess, model_dir + 'final_model.ckpt')
					else:
						print('iter {}\tvalid_loss = {:.6f}\t'.format(
							iter, valid_loss))
				iiter += 1

	print('stop training !!!')
