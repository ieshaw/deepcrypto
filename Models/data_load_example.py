__author__ = 'Ian'

import os
import pandas as pd
from Data.scripts.data import data

#set can be 'train', 'cross_val', or 'test'

X,Y = data.import_data(set= 'train')



print(X.head())

print(Y.head())