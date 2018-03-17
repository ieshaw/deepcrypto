__author__ = 'Ian'

import os
import time

import numpy as np
import pandas as pd
import torch
from torch.autograd import Variable

from Data.scripts.data import data
from Models.RNN.scripts.SimpleRNN import SimpleRNN
from Models.Evaluation.eval import eval_model

def run_model(params_file_string, hidden_size, X_df, Y_df, X_matrix, Y_matrix):
    # import data
    X, Y = X_df, Y_df
    x = X_matrix
    y = Y_matrix

    # set preditcion matrix
    y_pred = np.zeros(shape=y.shape)

    # set model
    model = SimpleRNN(hidden_size= hidden_size,input_size=len(X.iloc[0:1].values[0]), output_size=len(Y.iloc[0:1].values[0]))
    model.load_state_dict(torch.load(params_file_string))

    for iter in range(len(x)):
        input = Variable(torch.from_numpy(x[iter]).float())
        target = Variable(torch.from_numpy(y[iter]).float())

        output, hidden = model.forward(input)

        y_pred[iter] = output.data.numpy()

    Y_pred = pd.DataFrame(data=y_pred, index=Y.index, columns=Y.columns)

    return Y_pred


#load in data

X_train_df, Y_train_df = data.import_data(set='train')
X_train_matrix = X_train_df.as_matrix()
Y_train_matrix = Y_train_df.as_matrix()

X_dev_df, Y_dev_df = data.import_data(set='cross_val')
X_dev_matrix = X_dev_df.as_matrix()
Y_dev_matrix = Y_dev_df.as_matrix()

output_file = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))  + '/model_compare/model_compare.csv'

#loop through models in some directory

model_params_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))  + '/model_params/'

for file in os.listdir(model_params_dir):

    # out_dict = {'Model':, 'Max_d_train:': , 'Return_train': , 'Sharpe_train': , 'AccScore_train': ,'TN_train': ,
    # 'FP_train': , 'FN_train': , 'TP_train': ,'Max_d_train:': , 'Return_train': , 'Sharpe_train': , 'AccScore_train': ,'TN_train': ,
    # 'FP_train': , 'FN_train': , 'TP_train': }

    model_name = file.split('.')[0]

    hidden_size = model_name.split('_')[1]

    if hidden_size == 'parmas':
        hidden_size = 10
    else:
        hidden_size = int(hidden_size)

    model_params_file_str = model_params_dir + file

    Y_train_pred_df = run_model(params_file_string= model_params_file_str, hidden_size= hidden_size, X_df= X_train_df, Y_df= Y_train_df,
                             X_matrix= X_train_matrix, Y_matrix= Y_train_matrix)

    # run eval class

    check_model = eval_model(y_pred_df= Y_train_pred_df, y_actual_df= Y_train_df)

    check_model.backtest(printer=False)

    check_model.accuracy(printer=False)

    train_metrics_dict = check_model.metrics

    train_acc_score = check_model.accuracy_score

    train_conf_list = check_model.confusion_matrix

    # do for both train and dev set metrics

    Y_dev_pred_df = run_model(params_file_string= model_params_file_str,hidden_size= hidden_size, X_df= X_dev_df, Y_df= Y_dev_df,
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

    # Get the MSE Loss

    loss_df = pd.read_csv(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))  + '/loss_csvs/' + model_name + '.csv',
                          index_col= 0, header=0)

    loss = (loss_df.loc[loss_df.last_valid_index()])['loss']

    print('Model: {}. Has Loss: {}. Variance: {}.'.format(model_name, loss, variance))

    # write results to some csv

    with open(output_file, 'a') as f:
        # Model,Max_d_train,Return_train,Sharpe_train,AccScore_train,TN_train,FP_train,FN_train,TP_train,
        # Max_d_dev,Return_dev,Sharpe_dev,AccScroe_dev,TN_dev,FP_dev,FN_dev,TP_dev,loss,variance
        f.write('\n{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}'.format(
            model_name, train_metrics_dict['max_drawdown'], train_metrics_dict['return'], train_metrics_dict['sharpe'],
        train_acc_score, train_conf_list[0], train_conf_list[1], train_conf_list[2], train_conf_list[3],
        dev_metrics_dict['max_drawdown'], dev_metrics_dict['return'], dev_metrics_dict['sharpe'], dev_acc_score,
        dev_conf_list[0], dev_conf_list[1], dev_conf_list[2], dev_conf_list[3], loss, variance))












