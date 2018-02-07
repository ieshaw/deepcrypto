__author__ = 'Ian'

import os
import pandas as pd
from Data.scripts.data import data


X,Y = data.import_data(set= 'train')

print(X.head())

print(Y.head())