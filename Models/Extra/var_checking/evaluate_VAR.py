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


def fit_VAR(results, set_str):

    X_test, Y_test = data.import_data(set=set_str)

    # predict on test set
    predictions_test = np.zeros((X_test.shape[0], X_test.shape[1]))
    # turn into numpy array
    X_test_matrix = X_test.values
    # predict one-step ahead out-of-sample
    for i in range(0, X_test.shape[0]):
        predictions_test[i] = results.forecast(X_test_matrix[i, :].reshape(1, 20), steps=1)

    # Turn back into panda dataframe and save to csv
    Test_pred = pd.DataFrame(data=predictions_test, index=X_test.index, columns=X_test.columns)
    columns = ['XMRspread', 'XMRvolume', 'XMRbasevolume', 'XRPspread', 'XRPvolume', 'XRPbasevolume', 'LTCspread',
               'LTCvolume', 'LTCbasevolume', 'DASHspread', 'DASHvolume', 'DASHbasevolume', 'ETHspread', 'ETHvolume',
               'ETHbasevolume']
    Test_pred.drop(columns, 1, inplace=True)

    Y = Y_test

    Y_pred = Test_pred

    flat_pred = np.clip(Y_pred.as_matrix().flatten() + 0.5, 0, 1)

    flat_actual = np.where(Y.as_matrix().flatten() > 0, 1, 0)

    auc = roc_auc_score(flat_actual, flat_pred)

    mse = mean_squared_error(Y.as_matrix(), Y_pred.as_matrix())

    return Test_pred, auc, mse

X_train_df, Y_train_df = data.import_data(set='train')
X_train_matrix = X_train_df.as_matrix()
Y_train_matrix = Y_train_df.as_matrix()

X_dev_df, Y_dev_df = data.import_data(set='cross_val')
X_dev_matrix = X_dev_df.as_matrix()
Y_dev_matrix = Y_dev_df.as_matrix()

VAR_model = VAR(X_train_df)

results = VAR_model.fit(1)


Y_train_pred_df, train_auc, train_mse = fit_VAR(results, 'train')

Y_dev_pred_df, dev_auc, dev_mse = fit_VAR(results, 'cross_val')

model_name = 'VAR_int'


# run eval class

check_model = eval_model(y_pred_df= Y_train_pred_df, y_actual_df= Y_train_df)

check_model.backtest(printer=False)

check_model.accuracy(printer=False)

train_metrics_dict = check_model.metrics

train_acc_score = check_model.accuracy_score

train_conf_list = check_model.confusion_matrix

# do for both train and dev set metrics

# run eval class

check_model = eval_model(y_pred_df= Y_dev_pred_df, y_actual_df= Y_dev_df)

check_model.backtest(printer=False)

check_model.accuracy(printer=False)

dev_metrics_dict = check_model.metrics

dev_acc_score = check_model.accuracy_score

dev_conf_list = check_model.confusion_matrix

# get the variance: acc_score train - cross_val

variance = train_acc_score - dev_acc_score

# # Get the MSE Loss
#
# loss_df = pd.read_csv(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))  + '/loss_csvs/' + model_name + '.csv',
#                       index_col= 0, header=0)
#
# loss = (loss_df.loc[loss_df.last_valid_index()])['loss']

print('Train')
print(train_metrics_dict)
print('AUC: {} , MSE: {}'.format(train_auc, train_mse))
print('Acc: {}'.format(train_acc_score))
print('Dev')
print(dev_metrics_dict)
print('AUC: {} , MSE: {}'.format(dev_auc, dev_mse))
print('Acc: {}'.format(dev_acc_score))
print('Acc Var: {}'.format(train_acc_score - dev_acc_score))

filename = os.path.abspath(os.path.dirname(__file__)) + "/{}.txt".format(model_name)
with open(filename, 'w') as f:
    f.write('Train \n')
    f.write('{}\n'.format(train_metrics_dict))
    f.write('AUC: {} , MSE: {}\n'.format(train_auc, train_mse))
    f.write('Acc: {}\n'.format(train_acc_score))
    f.write('Dev\n')
    f.write('{}\n'.format(dev_metrics_dict))
    f.write('AUC: {} , MSE: {}\n'.format(dev_auc, dev_mse))
    f.write('Acc: {}\n'.format(dev_acc_score))
    f.write('Acc Var: {}\n'.format(train_acc_score - dev_acc_score))