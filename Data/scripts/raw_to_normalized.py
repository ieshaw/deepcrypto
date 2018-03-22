__author__ = 'Ian'

import os
import pandas as pd

def rolling_normal(series, length= 730, min_periods= 730):

    return (series - series.rolling(window= length, min_periods= min_periods).mean())/\
           series.rolling(window= length, min_periods= min_periods).std()

all_df = pd.read_csv(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..')) + '/csvs/all_raw.csv',
                     index_col= 0)

coins = ['ETH', 'XRP','LTC', 'DASH', 'XMR']

normal_df = pd.DataFrame(index= all_df.index)

for coin in coins:

    #create retruns

    # normal_df[coin+'return'] = (all_df['close'+coin] - all_df['open'+coin])/all_df['open'+coin]
    # normal_df[coin + 'return'] = (all_df['close' + coin] * (all_df['open' + coin].pow(-1))) - 1
    normal_df[coin + 'return'] = all_df['open' + coin].pct_change()

    #create spread

    normal_df[coin + 'spread'] = all_df['high' + coin] - all_df['low' + coin]

    #normalize spread

    normal_df[coin + 'spread'] = rolling_normal(normal_df[coin + 'spread'])

    #normalize volume

    normal_df[coin + 'volume'] = rolling_normal(all_df['volume' + coin])

    # normalize basevolume

    normal_df[coin + 'basevolume'] = rolling_normal(all_df['basevolume' + coin])

normal_df = normal_df.dropna()

normal_df.to_csv(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..')) + '/csvs/all_normal.csv')