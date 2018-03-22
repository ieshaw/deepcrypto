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
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR
from sklearn.metrics import mean_squared_error, roc_auc_score, roc_curve, auc


X,Y = data.import_data(set='test')

Y_pred = pd.read_csv('/Users/ianshaw/Downloads/GitHub/deepcrypto/Models/VAR/predicted_values_VAR_test_shifted.csv',
                     index_col = 0)


check_model = eval_model(y_pred_df= Y_pred, y_actual_df= Y)

check_model.backtest(printer=False)

check_model.accuracy(printer=False)

check_model.strat_series.plot()
plt.show()

train_metrics_dict = check_model.metrics

train_acc_score = check_model.accuracy_score

train_conf_list = check_model.confusion_matrix

print('Train')
print(train_metrics_dict)
print('Acc: {}'.format(train_acc_score))