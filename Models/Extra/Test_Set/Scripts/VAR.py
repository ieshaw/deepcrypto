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

# initialize predict on test set
predictions_test = np.zeros((X_test.shape[0],X_test.shape[1]))
predictions_test_stress = np.zeros((X_test.shape[0],X_test.shape[1]))

# turn into numpy array
X_test_matrix = X_test.values

# predict one-step ahead out-of-sample
for i in range(0,X_test.shape[0]):
    predictions_test[i] = results.forecast(X_test_matrix[i,:].reshape(1,20), steps=1)

# Turn back into panda data frame and save to csv
Test_pred = pd.DataFrame(data=predictions_test, index=X_test.index, columns=X_test.columns)
columns = ['XMRspread', 'XMRvolume', 'XMRbasevolume','XRPspread', 'XRPvolume', 'XRPbasevolume','LTCspread', 'LTCvolume', 'LTCbasevolume', 'DASHspread', 'DASHvolume', 'DASHbasevolume','ETHspread', 'ETHvolume', 'ETHbasevolume']
Test_pred.drop(columns, 1, inplace=True)
Test_pred.to_csv(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) + '/Pred_CSVs/VAR_extra.csv')