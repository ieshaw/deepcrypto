__author__ = 'Ian'

import matplotlib.pyplot as plt
import os
import pandas as pd
from Data.scripts.data import data

def time_series_plot(returns_series, ticker = 'BTC'):


    #returns_series.index = pd.to_datetime(returns_series.index, format='%Y-%m-%d %H:%M:%S')


    fig_hist = plt.figure()
    plt.title('Histogram of {0} {1} Returns'.format(ticker, 'Hourly'))
    returns_series.hist(bins= 100, log= True)

    #generate time series of returns

    fig_ts = plt.figure()
    plt.xlabel('Date')
    plt.ylabel('Returns')
    plt.title('Time Series of {0} {1} Returns'.format(ticker, 'Hourly'))
    ((returns_series + 1).cumprod()).plot(rot=45)

    #save results

    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) + '/Baseline/plots'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    fig_hist.savefig('{0}/{1}_hist.png'.format(output_dir, ticker))
    fig_ts.savefig('{0}/{1}_ts.png'.format(output_dir, ticker))

#set can be 'train', 'cross_val', or 'test'

X,Y = data.import_data(set= 'train')


coins = ['ETH', 'XRP','LTC', 'DASH', 'XMR']


for coin in coins:
    time_series_plot(returns_series= X[coin + 'return'], ticker= coin)