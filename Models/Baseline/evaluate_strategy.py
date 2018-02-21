__author__ = 'Ian'

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import pandas as pd
from Data.scripts.data import data

def run_strategy(Y_pred, Returns_df):

    # normalize rows to a sum of 1
    # sum the rows of the prediction, divide by that number

    Y_pred = Y_pred.divide(Y_pred.sum(axis=1), axis= 'index')

    strat_returns_series = (Y_pred.multiply(Returns_df + 1, axis= 'index')).sum(axis=1)

    #if at any point not invested in the market, hold value

    strat_returns_series = strat_returns_series.replace(to_replace = 0, value= 1)

    return strat_returns_series

    #evaluate the
X,Y = data.import_data(set= 'train')

coins = ['ETH', 'XRP','LTC', 'DASH', 'XMR']

returns_df = X[[coin + 'return' for coin in coins]]
returns_df.columns = coins

Y_ones = (Y*0) + 1



output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) + '/Baseline/plots'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

fig_ts = plt.figure()

strat_series = (run_strategy(Y_ones, returns_df)).cumprod()

strat_series.index = pd.to_datetime(strat_series.index, format='%Y-%m-%d %H:%M:%S')

strat_series.plot(rot= 45)
# plt.gcf().autofmt_xdate()
# myFmt = mdates.DateFormatter('%m/%Y')
# plt.gca().xaxis.set_major_formatter(myFmt)

# locator = mdates.MonthLocator()
# plt.gca().xaxis.set_major_locator(locator)
#
# plt.gcf().autofmt_xdate()
plt.xlabel('Date')
plt.ylabel('Returns')
plt.title('Time Series of Equal Investment Returns')

fig_ts.savefig('{0}/baseline_ts.png'.format(output_dir))