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

    if Y_pred.shape[0] < Returns_df.shape[0]:
        Returns_df = Returns_df.loc[Y_pred.index]
    else:
        Y_pred = Y_pred.loc[Returns_df.loc]

    #make negative returns 0 valued

    Y_pred = Y_pred.where(Y_pred > 6e-06, 0)

    # normalize rows to a sum of 1
    # sum the rows of the prediction, divide by that number

    Y_pred_sum = Y_pred.sum(axis=1)

    #if any sum is 0, meaning no investment, turn to 1 so as to avoid divide by 0

    Y_pred_sum = Y_pred_sum.where(Y_pred_sum > 1e-06, 1)

    Y_pred = Y_pred.divide(Y_pred_sum, axis= 'index')

    strat_returns_series = (Y_pred.multiply(Returns_df + 1, axis= 'index')).sum(axis=1)

    #if at any point not invested in the market, hold value

    strat_returns_series = strat_returns_series.replace(to_replace = 0, value= 1)

    return strat_returns_series

    #evaluate the

def strat_metrics(strat_series):

    metrics = {}

    metrics['return'] = strat_series[-1]

    risk_free = 0

    metrics['sharpe'] = ((strat_series[-1]-1) - risk_free)/(np.std(strat_series))

    metrics['max_drawdown'] = (1 - strat_series.div(strat_series.cummax())).max()

    return metrics


X,Y = data.import_data(set= 'train')

Y_pred = pd.read_csv(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) + '/csvs/y_pred.csv', index_col= 0)

Y_pred.columns = Y.columns

coins = ['ETH', 'XRP','LTC', 'DASH', 'XMR']

strat_series = (run_strategy(Y_pred= Y_pred, Returns_df= Y))

strat_series.plot()
plt.show()

strat_series = strat_series.cumprod()

strat_series.index = pd.to_datetime(strat_series.index, format='%Y-%m-%d %H:%M:%S')

print(strat_metrics(strat_series))

output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) + '/plots'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

fig_ts = plt.figure()

#Plotting
strat_series.plot(rot= 45)
plt.xlabel('Date')
plt.ylabel('Returns')
plt.title('Time Series of VAR Signal Returns')

fig_ts.savefig('{0}/var_ts.png'.format(output_dir))