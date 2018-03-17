__author__ = 'Ian'

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from Data.scripts.data import data
from pandas.tools.plotting import autocorrelation_plot

from Models.Evaluation.eval import eval_model

def get_file_names(model = 'VAR', y_pred_csv_name= 'y_pred', plot_suffix = '', loss_file_name = None):

    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))  + '/{}'.format(model)

    csv_dir = '{}/csvs'.format(model_path)

    y_pred_file = '{}/{}.csv'.format(csv_dir , y_pred_csv_name)

    plot_dir = '{}/plots'.format(model_path)

    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    ts_plot_path = '{}/ts_{}_{}.png'.format(plot_dir, model, plot_suffix)

    if loss_file_name is not None:

        loss_plot_path = '{}/loss_{}_{}.png'.format(plot_dir, model, plot_suffix)

        loss_file = '{}/{}.csv'.format(csv_dir, loss_file_name)

        return [y_pred_file, ts_plot_path, loss_file, loss_plot_path]

    else:

        return [y_pred_file, ts_plot_path]

#model options: VAR, RNN , R2N2, Baselime, Perfect
model = 'Perfect'
y_pred_csv_name = 'train_y'
plot_suffix = 'train'
loss_file_name = None

#set options: 'train', 'cross_val', or 'test'
set = 'train'

file_list = get_file_names(model= model, y_pred_csv_name= y_pred_csv_name,
                           plot_suffix= plot_suffix, loss_file_name= loss_file_name)

check_model = eval_model(y_pred_file= file_list[0], set= set)

check_model.backtest(printer=True)

check_model.plot_backtest(file_path=file_list[1])

if loss_file_name is not None:
    check_model.plot_loss(loss_csv_path= file_list[2], loss_plot_path= file_list[3])

check_model.accuracy(printer= True)
