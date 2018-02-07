__author__ = 'Ian'

import pandas as pd
import os

ratios = {'train': 0.7, 'cross_val': 0.85}

x_df = pd.read_csv(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..')) + '/csvs/all_normal.csv',
                   index_col=0)

y_df = pd.read_csv(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..')) + '/csvs/y.csv',
                   index_col=0)

#y doesnt have the last index that x does

x_df = x_df.drop(x_df.last_valid_index())

index_dict = {}

index_dict['train'] = x_df.iloc[0:int(ratios['train']*len(x_df))].index

index_dict['cross_val'] = x_df.iloc[int(ratios['train']*len(x_df)): int(ratios['cross_val']*len(x_df))].index

index_dict['test'] = x_df.iloc[int(ratios['cross_val']*len(x_df)):].index


for key in index_dict:

    output_dir = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..')) + '/csvs/{}'.format(key)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    x_df.loc[index_dict[key]].to_csv('{}/x.csv'.format(output_dir))

    y_df.loc[index_dict[key]].to_csv('{}/y.csv'.format(output_dir))


