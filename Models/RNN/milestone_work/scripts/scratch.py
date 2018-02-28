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

csv_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) + '/csvs'

loss_file = '{}/loss.csv'.format(csv_dir)

plot_output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) + '/plots'

ts_plot_path = '{0}/class_rnn_ts.png'.format(plot_output_dir)

loss_plot_path = '{0}/class_rnn_loss.png'.format(plot_output_dir)

#Y_pred = pd.read_csv(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) + '/csvs/y_pred.csv', index_col= 0)

y_pred_file = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) + '/csvs/y_pred.csv'

check_model = eval_model(y_pred_file= y_pred_file, set= 'train')

# check_model.backtest(printer=True)
#
# check_model.plot_backtest(file_path=ts_plot_path)
#
# check_model.plot_loss(loss_csv_path= loss_file, loss_plot_path= loss_plot_path)

check_model.accuracy(printer= True)
