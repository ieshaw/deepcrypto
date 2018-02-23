__author__ = 'Ian'

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from Data.scripts.data import data
from pandas.tools.plotting import autocorrelation_plot

X,Y = data.import_data(set= 'train')

coins = ['ETH', 'XRP','LTC', 'DASH', 'XMR']

Y_pred = (Y*0) + 1

Y_pred.to_csv(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) + '/csvs/y_pred.csv')