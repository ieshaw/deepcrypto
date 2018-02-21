__author__ = 'Ian'

import matplotlib.pyplot as plt
import os
import pandas as pd
from Data.scripts.data import data

X,Y = data.import_data(set= 'train')

print(X.index)

print((pd.to_datetime(X.index, format='%Y-%m-%d %H:%M:%S')).year)