__author__ = 'Ian'

#using https://gist.github.com/spro/ef26915065225df65c1187562eca7ec4

import os
import time

import numpy as np
import pandas as pd
import torch
from statsmodels.tsa.api import VAR
from torch.autograd import Variable

from Data.scripts.data import data
from Models.R2N2.Local.scripts.RNN import RNN
from Models.Evaluation.eval import eval_model
from sklearn.metrics import mean_squared_error, roc_auc_score, roc_curve, auc

def run_model(model_name, hidden_size):

    # import data
    # X, Y = data.import_data(set='cross_val')
    X, Y = data.import_data(set='train')

    # do not plug in returns, but residuals
    # plug in residuals
    VAR_model = VAR(X)

    results = VAR_model.fit(1)
    ar_returns = results.fittedvalues

    # columns to drop from dataframe
    columns = ['XMRspread', 'XMRvolume', 'XMRbasevolume', 'XRPspread', 'XRPvolume', 'XRPbasevolume', 'LTCspread',
               'LTCvolume', 'LTCbasevolume', 'DASHspread', 'DASHvolume', 'DASHbasevolume', 'ETHspread', 'ETHvolume',
               'ETHbasevolume']
    ar_returns.drop(columns, 1, inplace=True)

    X = X.loc[ar_returns.index]
    x_returns = X[ar_returns.columns]
    residual_df = x_returns - ar_returns
    X = X.join(residual_df, how='inner', rsuffix='residual')

    y_ar_returns = ar_returns
    y_ar_returns.columns = Y.columns
    Y = (Y.loc[X.index] - y_ar_returns.shift(-1)).dropna()
    y_ar_returns = y_ar_returns.shift(-1).dropna()
    X = X.loc[Y.index]

    x = X.as_matrix()
    y = Y.as_matrix()

    # set preditcion matrix
    y_pred = np.zeros(shape=y.shape)

    # set model
    model = RNN(hidden_size=hidden_size, input_size=len(X.iloc[0:1].values[0]), output_size=len(Y.iloc[0:1].values[0]))
    model.load_state_dict(
        torch.load(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) +
                   '/model_params/{}.pth.tar'.format(model_name)))

    for iter in range(len(x)):
        input = Variable(torch.from_numpy(x[iter]).float())

        output = model.forward(input)

        y_pred[iter] = output.data.numpy()

    y_pred = y_pred + y_ar_returns.as_matrix()

    Y_pred = pd.DataFrame(data=y_pred, index=Y.index, columns=Y.columns)

    flat_pred = np.clip(Y_pred.as_matrix().flatten() + 0.5, 0, 1)

    flat_actual = np.where(Y.as_matrix().flatten() > 0, 1, 0)

    auc = roc_auc_score(flat_actual, flat_pred)

    mse = mean_squared_error(Y.as_matrix(), Y_pred.as_matrix())

    return Y_pred, auc, mse


torch.manual_seed(1)

model_string = 'Mom_LSTM_6_BFC_1_AFC_1_Act_None'
hidden_size = 10
model_name = '{}_H{}'.format(model_string, hidden_size)

model_params_file_str = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) + '/model_params/{}.pth.tar'.format(model_name)

X_train_df, Y_train_df = data.import_data(set='train')
X_train_matrix = X_train_df.as_matrix()
Y_train_matrix = Y_train_df.as_matrix()

X_dev_df, Y_dev_df = data.import_data(set='cross_val')
X_dev_matrix = X_dev_df.as_matrix()
Y_dev_matrix = Y_dev_df.as_matrix()

Y_train_pred_df, train_auc, train_mse = run_model(model_name, hidden_size)

# run eval class

check_model = eval_model(y_pred_df= Y_train_pred_df, y_actual_df= Y_train_df)

check_model.backtest(printer=False)

check_model.accuracy(printer=False)

train_metrics_dict = check_model.metrics

train_acc_score = check_model.accuracy_score

train_conf_list = check_model.confusion_matrix


print('Train')
print(train_metrics_dict)
print('AUC: {} , MSE: {}'.format(train_auc, train_mse))
print('Acc: {}'.format(train_acc_score))


filename = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) + "/records/{}.txt".format(model_name)
with open(filename, 'w') as f:
    f.write('Train \n')
    f.write('{}\n'.format(train_metrics_dict))
    f.write('AUC: {} , MSE: {}\n'.format(train_auc, train_mse))
    f.write('Acc: {}\n'.format(train_acc_score))


