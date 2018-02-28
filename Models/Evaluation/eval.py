__author__ = 'Ian'

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from Data.scripts.data import data
from pandas.tools.plotting import autocorrelation_plot
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


def run_strategy(Y_pred, Y_actual):

    #make negative returns 0 valued

    Y_pred = Y_pred.where(Y_pred > 6e-06, 0)

    # normalize rows to a sum of 1
    # sum the rows of the prediction, divide by that number

    Y_pred_sum = Y_pred.sum(axis=1)

    #if any sum is 0, meaning no investment, turn to 1 so as to avoid divide by 0

    Y_pred_sum = Y_pred_sum.where(Y_pred_sum > 1e-06, 1)

    Y_pred = Y_pred.divide(Y_pred_sum, axis= 'index')

    strat_returns_series = (Y_pred.multiply(Y_actual + 1, axis= 'index')).sum(axis=1)

    #if at any point not invested in the market, hold value

    strat_returns_series = strat_returns_series.replace(to_replace = 0, value= 1)

    return strat_returns_series

def strat_metrics(strat_series):

    metrics = {}

    metrics['return'] = strat_series[-1]

    risk_free = 0

    metrics['sharpe'] = ( (strat_series[-1]-1) - risk_free)/(np.std(strat_series))

    metrics['max_drawdown'] = (1 - strat_series.div(strat_series.cummax())).max()

    return metrics

def returns_df_to_01(returns_df):

    for column in returns_df.columns:
        returns_df[column] = returns_df[column].where(returns_df[column] > 0, other = 0)

        returns_df[column] = returns_df[column].where(returns_df[column] == 0, other = 1)

    return returns_df.astype(int)

def find_accuracy(Y_pred, Y_actual):

    if Y_pred.shape[0] < Y_actual.shape[0]:
        Y_actual = Y_actual.loc[Y_pred.index]
    else:
        Y_pred = Y_pred.loc[Y_actual.index]

    Y_pred_01 = returns_df_to_01(returns_df= Y_pred).as_matrix().flatten()
    Y_actual_01 = returns_df_to_01(returns_df= Y_actual).as_matrix().flatten()

    return accuracy_score(y_true= Y_actual_01, y_pred= Y_pred_01)

def find_confusion_matrix(Y_pred, Y_actual):

    if Y_pred.shape[0] < Y_actual.shape[0]:
        Y_actual = Y_actual.loc[Y_pred.index]
    else:
        Y_pred = Y_pred.loc[Y_actual.index]

    Y_pred_01 = returns_df_to_01(returns_df= Y_pred).as_matrix().flatten()
    Y_actual_01 = returns_df_to_01(returns_df= Y_actual).as_matrix().flatten()

    return confusion_matrix(y_true= Y_actual_01, y_pred= Y_pred_01)

class eval_model:

    '''
    A class to evaluate models
    '''

    def __init__(self,y_pred_file, set= 'train'):

        self.y_pred = pd.read_csv(y_pred_file, index_col=0)

        _, self.y_actual = data.import_data(set=set)

        self.y_pred.columns = self.y_actual.columns


    def backtest(self, printer= True):

        self.strat_series = (run_strategy(Y_pred=self.y_pred, Y_actual=self.y_actual)).cumprod()

        self.strat_series.index = pd.to_datetime(self.strat_series.index, format='%Y-%m-%d %H:%M:%S')

        self.metrics = strat_metrics(self.strat_series)

        if printer:
            print(self.metrics)

    def plot_backtest(self, file_path):

        fig_ts = plt.figure()

        # Plotting
        self.strat_series.plot(rot=45)
        plt.xlabel('Date')
        plt.ylabel('Returns')
        plt.title('Time Series of Signal Returns')

        fig_ts.savefig(file_path)

    def plot_loss(self, loss_csv_path, loss_plot_path):

        loss_df = pd.read_csv(loss_csv_path, index_col=0)

        loss_series = loss_df['loss']

        fig_loss = plt.figure()
        loss_series.plot()
        # returns_series.plot(rot=45)
        # plt.gcf().autofmt_xdate()
        # myFmt = mdates.DateFormatter('%m/%Y')
        # plt.gca().xaxis.set_major_formatter(myFmt)

        # locator = mdates.MonthLocator()
        # plt.gca().xaxis.set_major_locator(locator)
        #
        # plt.gcf().autofmt_xdate()
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('MSE Loss by Epoch')

        fig_loss.savefig(loss_plot_path)

    def accuracy(self, printer= True):

        self.accuracy_score = find_accuracy(Y_pred= self.y_pred, Y_actual= self.y_actual)

        self.confusion_matrix = find_confusion_matrix(Y_pred= self.y_pred, Y_actual= self.y_actual).ravel()

        self.confusion_matrix = self.confusion_matrix/(self.confusion_matrix.sum())

        if printer:

            print('Accuracy Score (tp + tn)/(total): {}'.format(self.accuracy_score))

            print('tn, fp, fn, tp: {}'.format(self.confusion_matrix))





