import statsmodels as sm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from statsmodels.tsa.api import VARMAX
from Data.scripts.data import data

X,Y = data.import_data(set= 'train')
X_test,Y_test = data.import_data(set= 'test')

# optimal p is 7
#for p in range(1,10):
#    model = VARMAX(endog=Y, order=(p,0), exog=X)
#    results = model.fit(maxiter=30)
#    predictions = results.predict()
#    predictions=(predictions.shift(-1)).dropna()
#    if predictions.shape[0] != Y.shape[0]:
#        outcome = Y.tail(predictions.shape[0])
#    else:
#        outcome = Y
#    accuracy_matrix = outcome*predictions
#    accuracy_matrix.values
#    accuracy.append(np.sum(np.sum((accuracy_matrix > 0)))/accuracy_matrix.size)
endog_Y = X.drop(columns = ['XMRspread', 'XMRvolume', 'XMRbasevolume','XRPspread', 'XRPvolume', 'XRPbasevolume','LTCspread', 'LTCvolume', 'LTCbasevolume', 'DASHspread', 'DASHvolume', 'DASHbasevolume','ETHspread', 'ETHvolume', 'ETHbasevolume'])
exog_X = X.drop(columns = ['ETHreturn','XRPreturn','LTCreturn','DASHreturn','XMRreturn'])
p = 1
model = VARMAX(endog=endog_Y, order=(p,0), exog=exog_X)
results = model.fit(maxiter=0)
predictions = results.predict()
predictions=(predictions.shift(-1)).dropna()

if predictions.shape[0] != Y.shape[0]:
    outcome = endog_Y.tail(predictions.shape[0])
else:
    outcome = endog_Y

accuracy_matrix = outcome*predictions
accuracy_matrix.values
accuracy = (np.sum(np.sum((accuracy_matrix > 0.0)))/accuracy_matrix.size)

# turn into numpy array
endog_Y_test = X_test.drop(columns = ['XMRspread', 'XMRvolume', 'XMRbasevolume','XRPspread', 'XRPvolume', 'XRPbasevolume','LTCspread', 'LTCvolume', 'LTCbasevolume', 'DASHspread', 'DASHvolume', 'DASHbasevolume','ETHspread', 'ETHvolume', 'ETHbasevolume'])
exog_X_test = X_test.drop(columns = ['ETHreturn','XRPreturn','LTCreturn','DASHreturn','XMRreturn'])
X_test_matrix = exog_X_test.values
Y_test_matrix = endog_Y_test.values

# predict on test set
predictions_test = np.zeros(endog_Y_test.shape)

# predict one-step ahead out-of-sample
for i in range(0,X_test.shape[0]):
    predictions_test[i] = results.predict(start=i, end=i,exog=X_test_matrix,endog=Y_test_matrix)

Test_pred = pd.DataFrame(data=predictions_test, index=Y_test.index, columns=Y_test.columns)
Test_pred.to_csv(os.path.dirname(__file__) + '/predicted_values_VARMAX_test.csv')

#predictions.to_csv(os.path.dirname(__file__) + '/predicted_values_VARMAX.csv')
print('Accuracy: ' + str(accuracy))