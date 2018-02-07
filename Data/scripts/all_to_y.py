__author__ = 'Ian'

import os
import pandas as pd


normal_df = pd.read_csv(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..')) + '/csvs/all_normal.csv',
                        index_col=0)

coins = ['ETH', 'XRP','LTC', 'DASH', 'XMR']

shifted_normal_df = (normal_df.shift(-1)).dropna()

y_df = pd.DataFrame(index= shifted_normal_df.index)

for coin in coins:

    #create retruns

    shifted_normal_df[coin + 'return'] = shifted_normal_df[coin + 'return'].where(shifted_normal_df[coin + 'return'] > 0,
                                                                                  other = 0)

    shifted_normal_df[coin + 'return'] = shifted_normal_df[coin + 'return'].where(
        shifted_normal_df[coin + 'return'] == 0, other = 1)

    y_df[coin] = shifted_normal_df[coin + 'return']


y_df = y_df.astype(int)

y_df.to_csv(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..')) + '/csvs/y.csv')