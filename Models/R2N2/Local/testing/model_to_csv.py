__author__ = 'Ian'

import os
import time

import numpy as np
import pandas as pd
import torch
from statsmodels.tsa.api import VAR
from torch.autograd import Variable

from Data.scripts.data import data
from Models.R2N2.Local.scripts.RNN import RNN
from Models.Evaluation.eval import eval_model

model_string = 'R2N2_ETRA'

model_params_file_str = '/Users/ianshaw/Downloads/GitHub/deepcrypto/Models/R2N2/Local/model_params/EXTRA_LSTM_6_BFC_1_AFC_1_Act_None_H10.pth.tar'


X, Y = data.import_data(set='cross_val')

# do not plug in returns, but residuals
# plug in residuals
VAR_model = VAR(X)

results = VAR_model.fit(1)
ar_returns = results.fittedvalues

# columns to drop from dataframe
columns = ['XMRspread', 'XMRvolume', 'XMRbasevolume', 'XRPspread', 'XRPvolume', 'XRPbasevolume', 'LTCspread',
           'LTCvolume', 'LTCbasevolume', 'DASHspread', 'DASHvolume', 'DASHbasevolume', 'ETHspread', 'ETHvolume',
           'ETHbasevolume']
ar_returns.drop(columns, 1, inplace=True)

X = X.loc[ar_returns.index]
x_returns = X[ar_returns.columns]
residual_df = x_returns - ar_returns
X = X.join(residual_df, how='inner', rsuffix='residual')

y_ar_returns = ar_returns
y_ar_returns.columns = Y.columns
Y = (Y.loc[X.index] - y_ar_returns.shift(-1)).dropna()
y_ar_returns = y_ar_returns.shift(-1).dropna()
X = X.loc[Y.index]

x = X.as_matrix()
y = Y.as_matrix()

# set model
model = RNN(hidden_size=10, input_size=len(X.iloc[0:1].values[0]), output_size=len(Y.iloc[0:1].values[0]))
model.load_state_dict(torch.load(model_params_file_str))

#run it through the cross val to develop memory

for iter in range(len(x)):
    input = Variable(torch.from_numpy(x[iter]).float())
    target = Variable(torch.from_numpy(y[iter]).float())

    output = model.forward(input)

#run it through the test to output csv

X, Y = data.import_data(set='test')

residual_df = pd.read_csv('/Users/ianshaw/Downloads/GitHub/deepcrypto/Models/Extra/Test_Set/Pred_CSVs/VAR_extra.csv', index_col=0)
ar_df = pd.read_csv('/Users/ianshaw/Downloads/GitHub/deepcrypto/Models/Extra/Test_Set/Pred_CSVs/VAR_extra.csv', index_col=0)
x_returns = X[['ETHreturn', 'XRPreturn', 'LTCreturn', 'DASHreturn', 'XMRreturn']]
residual_df = x_returns - residual_df
X = X.join(residual_df, how='inner', rsuffix='residual')

x = X.as_matrix()
y = Y.as_matrix()

#
# set preditcion matrix
y_pred = np.zeros(shape=y.shape)

for iter in range(len(x)):
    input = Variable(torch.from_numpy(x[iter]).float())

    output = model.forward(input)

    y_pred[iter] = output.data.numpy()

# Y_pred = pd.DataFrame(data=y_pred, index=Y.index, columns=Y.columns)

Y_pred = pd.DataFrame(data=y_pred + ar_df.as_matrix(), index=Y.index, columns=Y.columns)

Y_pred.to_csv('{}_test.csv'.format(model_string))