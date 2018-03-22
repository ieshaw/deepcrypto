__author__ = 'Ian'


import os
import pandas as pd

def f(x):
    d = {}

    #the individual functions for each
    d['high'] = x[' [High]'].max()
    d['low'] = x[' [Low]'].min()
    d['open'] = x[' [Open]'].iloc[0]
    d['close'] = x[' [Close]'][-1]
    d['volume'] = x[' [Volume]'].sum()
    d['basevolume'] = x[' [BaseVolume]'].sum()

    return pd.Series(d, index=['high', 'low', 'open', 'close', 'volume', 'basevolume'])

def min_filter_data_df(data_df):

    data_df = data_df.groupby(pd.TimeGrouper(freq='60s')).first()

    data_df[[u' [Open]', u' [Close]', u' [High]', u' [Low]',]] = \
        data_df[[u' [Open]', u' [Close]', u' [High]', u' [Low]',]].fillna(method='pad')

    data_df[[u' [Volume]',u' [BaseVolume]']] = data_df[[u' [Volume]',u' [BaseVolume]']].fillna(0)

    return data_df

def filter_data_df(data_df, filter_freq= '1H'):

    data_df = data_df.groupby(pd.TimeGrouper(freq=filter_freq, label='right')).apply(f)

    data_df = data_df.fillna(0)

    return data_df

def import_and_filter(data_file):

    historic_frame = pd.read_csv(data_file, skiprows=0, header=1)

    # make the index the timestamp
    historic_frame.index = pd.to_datetime(historic_frame['[TimeStamp]'], format='%m/%d/%Y %I:%M:%S %p')

    historic_frame = historic_frame.drop('[TimeStamp]', axis = 1)

    # fill in the minutes
    historic_frame = min_filter_data_df(historic_frame)

    # filter by hours

    historic_frame = filter_data_df(historic_frame)

    return historic_frame


#Import all the csvs to data frames

data_dir = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..'))

coins = ['ETH', 'XRP','LTC', 'DASH', 'XMR']

coin_df_dict = {}

for coin in coins:

    coin_df_dict[coin] = import_and_filter("{}/csvs/original_csv/BTC-{}.csv".format(data_dir,coin))

#inner join by data

all_df = coin_df_dict[coins[0]].join(coin_df_dict[coins[1]], how='inner', lsuffix=coins[0], rsuffix=coins[1])

for i in range(2,len(coins)):

    coin_df_dict[coins[i]].columns = [column + coins[i] for column in coin_df_dict[coins[i]].columns]

    all_df = all_df.join(coin_df_dict[coins[i]], how='inner')

all_df = all_df.dropna()

all_df.to_csv('{}/csvs/all_raw.csv'.format(data_dir))





