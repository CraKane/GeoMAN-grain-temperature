"""
@project = Predict the value according the data_geoMAN
@file = create_ckpt
@author = 10374
@create_time = 2019/11/26 3:32
"""

import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
import pickle
from utils import split_labels
import xlrd as xld


if __name__ == '__main__':
	# open the xlsx workbook
	workbook = xld.open_workbook("./graintem_1.xlsx")
	worksheet = workbook.sheet_by_name("温度值")
	col_data = worksheet.col_values(110)  # the first column
	col_data = col_data[1339:]
	# change it to array & reshape it
	col_data = np.array(col_data)
	col_data = col_data.reshape(-1, 1)
	print(worksheet)  # <xlrd.sheet.Sheet object at 0x000001B98D99CFD0>
	split_labels(col_data)
	# f = open('./scalers/scaler-0.pkl', 'rb')
	# scaler = pickle.load(f)
	# X = scaler.transform(col_data)
	# print(X)
	# f.close()