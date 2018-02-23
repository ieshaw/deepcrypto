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

#find optimal order of VAR model
results_optimal_order = VAR_model.fit(100)
predictions_optimal = results_optimal_order.fittedvalues

#columns to drop from dataframe
predictions_optimal.drop(columns, 1, inplace=True)
predictions_optimal=(predictions_optimal.shift(-1)).dropna()
predictions_optimal.to_csv(os.path.dirname(__file__) + '/predicted_values_VAR_optimal.csv')

print(predictions.dtypes)
print(predictions.head())
print(predictions_optimal.head())
