__author__ = 'Ian'

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from Data.scripts.data import data
from pandas.tools.plotting import autocorrelation_plot
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

def returns_df_to_01(returns_df):

    for column in returns_df.columns:
        returns_df[column] = returns_df[column].where(returns_df[column] > 0, other = 0)

        returns_df[column] = returns_df[column].where(returns_df[column] == 0, other = 1)

    return returns_df.astype(int)

def find_accuracy(Y_pred, Y_actual):

    if Y_pred.shape[0] < Y_actual.shape[0]:
        Y_actual = Y_actual.loc[Y_pred.index]
    else:
        Y_pred = Y_pred.loc[Y_actual.index]

    Y_pred_01 = returns_df_to_01(returns_df= Y_pred).as_matrix().flatten()
    Y_actual_01 = returns_df_to_01(returns_df= Y_actual).as_matrix().flatten()

    return accuracy_score(y_true= Y_actual_01, y_pred= Y_pred_01)

def find_confusion_matrix(Y_pred, Y_actual):

    if Y_pred.shape[0] < Y_actual.shape[0]:
        Y_actual = Y_actual.loc[Y_pred.index]
    else:
        Y_pred = Y_pred.loc[Y_actual.index]

    Y_pred_01 = returns_df_to_01(returns_df= Y_pred).as_matrix().flatten()
    Y_actual_01 = returns_df_to_01(returns_df= Y_actual).as_matrix().flatten()

    return confusion_matrix(y_true= Y_actual_01, y_pred= Y_pred_01)

X,Y = data.import_data(set= 'train')

Y_pred = pd.read_csv(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) + '/csvs/y_pred.csv', index_col= 0)

Y_pred.columns = Y.columns



print(find_accuracy(Y_pred= Y_pred, Y_actual= Y))

print(find_confusion_matrix(Y_pred= Y_pred, Y_actual= Y))