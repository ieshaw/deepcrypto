import statsmodels as sm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from statsmodels.tsa.api import VAR
from Data.scripts.data import data

X,Y = data.import_data(set= 'train')

VAR_model = VAR(X)

results = VAR_model.fit(1)
predictions = results.fittedvalues
#save the 1-order VAR model
results.save("One_order_VARmodel.pickle")

#columns to drop from dataframe
columns = ['XMRspread', 'XMRvolume', 'XMRbasevolume','XRPspread', 'XRPvolume', 'XRPbasevolume','LTCspread', 'LTCvolume', 'LTCbasevolume', 'DASHspread', 'DASHvolume', 'DASHbasevolume','ETHspread', 'ETHvolume', 'ETHbasevolume']
predictions.drop(columns, 1, inplace=True)
predictions=(predictions.shift(-1)).dropna() # shape = (10936, 5)
predictions.to_csv(os.path.dirname(__file__) + '/predicted_values_VAR.csv')

#drop last row of Y since the prediction and outcome are shifted
outcome = Y.drop(['2015-09-13 20:00:00','2016-12-12 13:00:00'],0) # shape = (10936, 5)

print(predictions.dtypes)
print(predictions.head())
#print(predictions.shape)
