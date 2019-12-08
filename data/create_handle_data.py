"""
@project = Predict the value according the data_geoMAN
@file = create_handle_data
@author = 10374
@create_time = 2019/11/26 3:32
"""

import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
import pickle
from utils import handle_data
import xlrd as xld
from sklearn import preprocessing


if __name__ == '__main__':
	# read input data
	# open the xlsx workbook
	workbook_grain = xld.open_workbook("./graintem_1.xlsx")
	worksheet_grain = workbook_grain.sheet_by_name("温度值")
	# read the local input
	local_index = [85, 86, 87, 89, 90, 91, 93, 94, 95, 105, 106,
				   107, 109, 111, 113, 114, 115, 125, 126, 127, 129,
				   130, 131, 133, 134, 135]
	local_input = []; local_inputs = []
	# read the data oneline by oneline
	for i in range(len(local_index)):
		col_data = worksheet_grain.col_values(local_index[i])  # the i-th column
		col_data = np.array(col_data[1:]).reshape(-1, 1)
		scaler = preprocessing.StandardScaler().fit(col_data)
		col_data = np.array(scaler.transform(col_data)).reshape(1, -1) # normalization
		local_input.insert(i, col_data[0])

	local_input = np.array(local_input)
	local_input = local_input.T
	# split the data to (?, 12, 26)
	for i in range(1338):
		local_inputs.insert(i, local_input[i:i+12])
	local_inputs = np.array(local_inputs)
	print(local_inputs)
	print(len(local_inputs[0]))
	print(len(local_inputs[0][0]))
	print("finished local data!")

	# read externel data
	workbook = xld.open_workbook("./mete.xls")
	worksheet = workbook.sheet_by_name("Sheet1")
	# read the externel input
	externel_input = []; externel_inputs = []
	nclos = worksheet.ncols
	for i in range(0, nclos-1):
		col_data = worksheet.col_values(i+1)  # the i-th column
		col_data = [float(x) for x in col_data[105:1448]]
		col_data = np.array(col_data).reshape(-1, 1)
		scaler = preprocessing.StandardScaler().fit(col_data)
		col_data = np.array(scaler.transform(col_data)).reshape(1, -1)[0]
		externel_input.insert(i, col_data)
	externel_input = np.array(externel_input)
	externel_input = externel_input.T
	# split the data to (?, 6, 8)
	for i in range(1338):
		externel_inputs.insert(i, externel_input[i:i + 6])
	externel_inputs = np.array(externel_inputs)
	print(externel_inputs)
	print("finished externel data!")

	# read label
	label_inputs = []
	col_data = worksheet_grain.col_values(110)  # the i-th column
	col_data = np.array(col_data[1:]).reshape(-1, 1)
	scaler = preprocessing.StandardScaler().fit(col_data)
	col_data = np.array(scaler.transform(col_data)).reshape(1, -1)[0]
	# split the data to (?, 6, 1)
	for i in range(1338):
		label_inputs.insert(i, col_data[i:i + 6])
	label_inputs = np.array(label_inputs)
	print(label_inputs)
	print(len(label_inputs[0]))
	print("finished label!")

	# read the global data
	global_input = []; global_attn_input = []; global_attn_inputs = []
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
		col_data = worksheet_grain.col_values(global_index[i]+1)  # the i-th column
		col_data = np.array(col_data[1:]).reshape(-1, 1)
		scaler = preprocessing.StandardScaler().fit(col_data)
		col_data = np.array(scaler.transform(col_data)).reshape(1, -1)
		global_input.insert(i, col_data[0])
	global_input = np.array(global_input)
	# split the data to (?, 174)
	for i in range(148):
		global_attn_input.insert(i, global_input[i:i + 26])
	for i in range(148, 174):
		global_attn_input.insert(i, global_input[148:174])
	global_attn_input = np.array(global_attn_input)
	# split the data to (?, 174, 26, 12)
	for j in range(1338):
		global_input_174 = []
		for i in range(174):
			tmp = global_attn_input[i,:,j:j+12]
			global_input_174.insert(i, tmp)
		global_input_174 = np.array(global_input_174)
		global_attn_inputs.insert(j, global_input_174)

	global_input = global_input.T
	global_attn_inputs = np.array(global_attn_inputs)
	print(global_attn_inputs)
	print(global_input)
	print(len(global_attn_inputs[0][0]))
	print(len(global_attn_inputs[0][0][0]))
	print("finished globel data!")
	# handle_data(local_inputs, externel_inputs, label_inputs, global_input, global_attn_inputs)
