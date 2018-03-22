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

X_test, Y_test = data.import_data(set='test')

# Y_pred_df = Y_test * 0
#
# Y_pred_df['ETH'] =  Y_pred_df['ETH'] + 1
#
# check_model = eval_model(y_pred_df= Y_pred_df, y_actual_df= Y_test)
#
# check_model.backtest(printer=False)
#
# check_model.accuracy(printer=False)
#
# dev_metrics_dict = check_model.metrics
#
# dev_acc_score = check_model.accuracy_score
#
# dev_conf_list = check_model.confusion_matrix
#
# print('Dev')
# print(dev_metrics_dict)
# print('Acc: {}'.format(dev_acc_score))
#
# print((Y_pred_df * Y_test + 1).prod())
#
# print(np.exp((np.log(Y_test['ETH'] + 1)).sum()) - 1)
#
# print(Y_test.first_valid_index())

Y_pred_df = (Y_test * 0) + 1

check_model = eval_model(y_pred_df= Y_pred_df, y_actual_df= Y_test)

check_model.backtest(printer=False)

check_model.accuracy(printer=False)

dev_metrics_dict = check_model.metrics

dev_acc_score = check_model.accuracy_score

dev_conf_list = check_model.confusion_matrix

print('Dev')
print(dev_metrics_dict)

print(((Y_test/5).sum(axis = 1) + 1).prod()-1)