__author__ = 'Ian'

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from Data.scripts.data import data
from pandas.tools.plotting import autocorrelation_plot

def run_strategy(Y_pred, Returns_df):

    # normalize rows to a sum of 1
    # sum the rows of the prediction, divide by that number

    Y_pred = Y_pred.divide(Y_pred.sum(axis=1), axis= 'index')

    strat_returns_series = (Y_pred.multiply(Returns_df + 1, axis= 'index')).sum(axis=1)

    #if at any point not invested in the market, hold value

    strat_returns_series = strat_returns_series.replace(to_replace = 0, value= 1)

    return strat_returns_series

    #evaluate the

def strat_metrics(strat_series):

    metrics = {}

    metrics['return'] = strat_series[-1]

    risk_free = 0

    metrics['sharpe'] = ( (strat_series[-1]-1) - risk_free)/(np.std(strat_series))

    metrics['max_drawdown'] = (1 - strat_series.div(strat_series.cummax())).max()

    return metrics


X,Y = data.import_data(set= 'train')

coins = ['ETH', 'XRP','LTC', 'DASH', 'XMR']

returns_df = X[[coin + 'return' for coin in coins]]
returns_df.columns = coins

# autocorrelation_plot(returns_df[coins[3])
#
# plt.show()
# Y_ones = (Y*0) + 1
#
# # print(Y.sum().sum()/Y_ones.sum().sum())
# #
# # print('Accuracy: {}'.format(accuracy_score(Y.values, Y_ones.values)))
#
# output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) + '/Baseline/plots'
# if not os.path.exists(output_dir):
#     os.makedirs(output_dir)
#
# fig_ts = plt.figure()
#
# strat_series = (run_strategy(Y_ones, returns_df)).cumprod()
#
# strat_series.index = pd.to_datetime(strat_series.index, format='%Y-%m-%d %H:%M:%S')
#
# print(strat_metrics(strat_series))

##Plotting
# strat_series.plot(rot= 45)
# plt.xlabel('Date')
# plt.ylabel('Returns')
# plt.title('Time Series of Equal Investment Returns')
#
# fig_ts.savefig('{0}/baseline_ts.png'.format(output_dir))