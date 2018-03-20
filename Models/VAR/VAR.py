import statsmodels as sm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from statsmodels.tsa.api import VAR
from Data.scripts.data import data

X,Y = data.import_data(set= 'train')
X_test,Y_test = data.import_data(set= 'test')

VAR_model = VAR(X)

results = VAR_model.fit(1)
predictions = results.fittedvalues

#save the 1-order VAR model
results.save("One_order_VARmodel.pickle")

# initialize predict on test set
predictions_test = np.zeros((X_test.shape[0],X_test.shape[1]))
predictions_test_stress = np.zeros((X_test.shape[0],X_test.shape[1]))

# turn into numpy array
X_test_matrix = X_test.values

# predict one-step ahead out-of-sample
for i in range(0,X_test.shape[0]):
    predictions_test[i] = results.forecast(X_test_matrix[i,:].reshape(1,20), steps=1)

# stress test for VAR
for i in range(0,int(X_test.shape[0]/2)):
    predictions_test_stress[i] = results.forecast(X_test_matrix[i,:].reshape(1,20), steps=1)
for i in range(int(X_test.shape[0]/2),X_test.shape[0]):
    predictions_test_stress[i] = results.forecast(np.random.randn(1,20), steps=1)

# Turn back into panda data frame and save to csv
Test_pred = pd.DataFrame(data=predictions_test, index=X_test.index, columns=X_test.columns)
columns = ['XMRspread', 'XMRvolume', 'XMRbasevolume','XRPspread', 'XRPvolume', 'XRPbasevolume','LTCspread', 'LTCvolume', 'LTCbasevolume', 'DASHspread', 'DASHvolume', 'DASHbasevolume','ETHspread', 'ETHvolume', 'ETHbasevolume']
Test_pred.drop(columns, 1, inplace=True)
Test_pred.to_csv(os.path.dirname(__file__) + '/predicted_values_VAR_test.csv')

# shifted data frame
Test_pred_shifted=(Test_pred.shift(-1)).dropna()
Test_pred_shifted.to_csv(os.path.dirname(__file__) + '/predicted_values_VAR_test_shifted.csv')

# stress test into panda data frame and csv
Stress_test = pd.DataFrame(data=predictions_test_stress, index=X_test.index, columns=X_test.columns)
Stress_test.drop(columns, 1, inplace=True)
Stress_test.to_csv(os.path.dirname(__file__) + '/predicted_values_VAR_test_stress.csv')

#columns to drop from dataframe
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