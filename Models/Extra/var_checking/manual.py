__author__ = 'Ian'

import os
import time

import numpy as np
import pandas as pd
import torch
from torch.autograd import Variable

from Data.scripts.data import data
from Models.Extra.scripts.RNN import RNN
from Models.Evaluation.eval import eval_model
from statsmodels.tsa.api import VAR
from sklearn.metrics import mean_squared_error, roc_auc_score, roc_curve, auc


X_train_df, Y_train_df = data.import_data(set='train')
X_train_matrix = X_train_df.as_matrix()
Y_train_matrix = Y_train_df.as_matrix()

#use just the returns, no other data
columns = ['XMRspread', 'XMRvolume', 'XMRbasevolume', 'XRPspread', 'XRPvolume', 'XRPbasevolume', 'LTCspread',
           'LTCvolume', 'LTCbasevolume', 'DASHspread', 'DASHvolume', 'DASHbasevolume', 'ETHspread', 'ETHvolume',
           'ETHbasevolume']
X_train_df.drop(columns, 1, inplace=True)

VAR_model = VAR(X_train_df)

results = VAR_model.fit(1)

print(results.summary())

# ret_cols = [0, 4, 8, 12, 16]
#
# VAR_mat = results.coefs[:, ret_cols]

VAR_mat = results.coefs

VAR_mat = np.transpose(VAR_mat.reshape(VAR_mat.shape[1], VAR_mat.shape[2]))

X_test, Y_test = data.import_data(set='train')

columns = ['XMRspread', 'XMRvolume', 'XMRbasevolume', 'XRPspread', 'XRPvolume', 'XRPbasevolume', 'LTCspread',
           'LTCvolume', 'LTCbasevolume', 'DASHspread', 'DASHvolume', 'DASHbasevolume', 'ETHspread', 'ETHvolume',
           'ETHbasevolume']
X_test.drop(columns, 1, inplace=True)

Y_pred = np.dot(X_test.as_matrix(), VAR_mat)

# Y_pred = Y_pred + np.tile(results.intercept[ret_cols], (Y_pred.shape[0], 1))
Y_pred = Y_pred + np.tile(results.intercept, (Y_pred.shape[0], 1))

Y_pred_df = pd.DataFrame(data=Y_pred, index= Y_test.index, columns= Y_test.columns)

check_model = eval_model(y_pred_df= Y_pred_df, y_actual_df= Y_test)

check_model.backtest(printer=False)

check_model.accuracy(printer=False)

dev_metrics_dict = check_model.metrics

dev_acc_score = check_model.accuracy_score

dev_conf_list = check_model.confusion_matrix

print('Dev')
print(dev_metrics_dict)
print('Acc: {}'.format(dev_acc_score))