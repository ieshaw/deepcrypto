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
from sklearn.metrics import mean_squared_error, roc_auc_score, roc_curve, auc

def run_model(params_file_string, hidden_size, X_df, Y_df, X_matrix, Y_matrix):
    # import data
    X, Y = X_df, Y_df
    x = X_matrix
    y = Y_matrix

    # set preditcion matrix
    y_pred = np.zeros(shape=y.shape)

    # set model
    # model = SimpleRNN(hidden_size= hidden_size,input_size=len(X.iloc[0:1].values[0]), output_size=len(Y.iloc[0:1].values[0]))
    model = RNN(hidden_size=hidden_size, input_size=len(X.iloc[0:1].values[0]),
                     output_size=len(Y.iloc[0:1].values[0]))
    model.load_state_dict(torch.load(params_file_string))

    for iter in range(len(x)):
        input = Variable(torch.from_numpy(x[iter]).float())
        target = Variable(torch.from_numpy(y[iter]).float())

        output = model.forward(input)

        y_pred[iter] = output.data.numpy()

    Y_pred = pd.DataFrame(data=y_pred, index=Y.index, columns=Y.columns)

    flat_pred = np.clip(Y_pred.as_matrix().flatten() + 0.5, 0, 1)

    flat_actual = np.where(Y.as_matrix().flatten() > 0, 1, 0)

    auc = roc_auc_score(flat_actual, flat_pred)

    mse = mean_squared_error(Y.as_matrix(), Y_pred.as_matrix())

    return Y_pred, auc, mse

#load in data

X_train_df, Y_train_df = data.import_data(set='train')
X_train_matrix = X_train_df.as_matrix()
Y_train_matrix = Y_train_df.as_matrix()

X_dev_df, Y_dev_df = data.import_data(set='cross_val')
X_dev_matrix = X_dev_df.as_matrix()
Y_dev_matrix = Y_dev_df.as_matrix()


model_string = 'LSTM_1_BFC_0_AFC_1_Act_None_LR_10_LRS_5_Epcoh_100'
hidden_size = 10
model_name = '{}_H{}'.format(model_string, hidden_size)

model_params_file_str = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) + '/model_params/{}.pth.tar'.format(model_name)

Y_train_pred_df, train_auc, train_mse = run_model(params_file_string= model_params_file_str, hidden_size= hidden_size, X_df= X_train_df, Y_df= Y_train_df,
                         X_matrix= X_train_matrix, Y_matrix= Y_train_matrix)

# run eval class

check_model = eval_model(y_pred_df= Y_train_pred_df, y_actual_df= Y_train_df)

check_model.backtest(printer=False)

check_model.accuracy(printer=False)

train_metrics_dict = check_model.metrics

train_acc_score = check_model.accuracy_score

train_conf_list = check_model.confusion_matrix

# do for both train and dev set metrics

Y_dev_pred_df, dev_auc, dev_mse = run_model(params_file_string= model_params_file_str,hidden_size= hidden_size, X_df= X_dev_df, Y_df= Y_dev_df,
                         X_matrix= X_dev_matrix, Y_matrix= Y_dev_matrix)

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

filename = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) + "/records/{}.txt".format(model_name)
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













