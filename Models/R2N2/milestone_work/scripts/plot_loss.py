__author__ = 'Ian'

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import pandas as pd
from Data.scripts.data import data

def loss_plot(loss_series):

    #generate time series of returns

    fig_ts = plt.figure()
    loss_series.plot()
    #returns_series.plot(rot=45)
    # plt.gcf().autofmt_xdate()
    # myFmt = mdates.DateFormatter('%m/%Y')
    # plt.gca().xaxis.set_major_formatter(myFmt)

    # locator = mdates.MonthLocator()
    # plt.gca().xaxis.set_major_locator(locator)
    #
    # plt.gcf().autofmt_xdate()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('MSE Loss by Epoch for R2N2')

    #save results

    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) + '/plots'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    fig_ts.savefig('{0}/r2n2_loss.png'.format(output_dir))

csv_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) + '/csvs'
loss_file = '{}/loss.csv'.format(csv_dir)
loss_df = pd.read_csv(loss_file, index_col= 0)

loss_plot(loss_series=loss_df['loss'])