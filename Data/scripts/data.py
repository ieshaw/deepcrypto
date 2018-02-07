__author__ = 'Ian'

import os
import pandas as pd

class data:

    '''
    A class in order to import data into any backtesting script
    '''

    def import_data(set= 'train'):

        '''
        :param set: 'train', 'cross_val', or 'test'
        :return: data_set, labels. as pandas dataframes
        '''

        x_df = pd.read_csv(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) + '/csvs/{}/x.csv'.format(set),
                                index_col=0)

        y_df = pd.read_csv(
            os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) + '/csvs/{}/y.csv'.format(set),
            index_col=0)

        return x_df, y_df