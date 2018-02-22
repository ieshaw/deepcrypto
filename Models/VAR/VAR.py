import statsmodels as sm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR
from Data.scripts.data import data

X,Y = data.import_data(set= 'train')

VAR_model = VAR(X)

results = VAR_model.fit(1)
predictions = results.fittedvalues

#columns to drop from dataframe
columns = ['XMRspread', 'XMRvolume', 'XMRbasevolume','XRPspread', 'XRPvolume', 'XRPbasevolume','LTCspread', 'LTCvolume', 'LTCbasevolume', 'DASHspread', 'DASHvolume', 'DASHbasevolume','ETHspread', 'ETHvolume', 'ETHbasevolume']
predictions.drop(columns, 1, inplace=True)
#make return predictions binary
predictions[predictions<0] = 0
predictions[predictions>0] = 1

#drop last row of Y since the prediction and outcome are shifted
outcome = Y.drop(['2016-12-12 13:00:00'],0)

print(predictions.dtypes)
print(predictions.head())
print(Y.head())
