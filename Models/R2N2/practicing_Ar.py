__author__ = 'Ian'

#using https://gist.github.com/spro/ef26915065225df65c1187562eca7ec4

import numpy as np
import os
import pandas as pd
import time
import torch
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math, random
from statsmodels.tsa.api import VAR

from Data.scripts.data import data
from Models.RNN.SimpleRNN import SimpleRNN


torch.manual_seed(1)

#import data
X,Y = data.import_data(set= 'train')

#do not plug in returns, but residuals
#plug in residuals
VAR_model = VAR(X)

results = VAR_model.fit(1)
ar_returns = results.fittedvalues

#columns to drop from dataframe
columns = ['XMRspread', 'XMRvolume', 'XMRbasevolume','XRPspread', 'XRPvolume', 'XRPbasevolume','LTCspread', 'LTCvolume', 'LTCbasevolume', 'DASHspread', 'DASHvolume', 'DASHbasevolume','ETHspread', 'ETHvolume', 'ETHbasevolume']
ar_returns.drop(columns, 1, inplace=True)

X = X.loc[ar_returns.index]
x_returns = X[ar_returns.columns]
residual_df = x_returns - ar_returns
X = X.join(residual_df, how = 'inner', rsuffix = 'residual')

y_ar_returns = ar_returns
y_ar_returns.columns = Y.columns
Y = (Y.loc[X.index] - y_ar_returns.shift(-1)).dropna()
y_ar_returns = y_ar_returns.shift(-1).dropna()

X = X.loc[Y.index]


print(X.shape)
print(y_ar_returns.shape)