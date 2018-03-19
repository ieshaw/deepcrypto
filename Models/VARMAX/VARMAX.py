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

p = 7
model = VARMAX(endog=Y, order=(p,0), exog=X)
results = model.fit(maxiter=0)
predictions = results.predict()
predictions=(predictions.shift(-1)).dropna()

#results.save("Optimal_order_VARXmodel.pickle")

if predictions.shape[0] != Y.shape[0]:
    outcome = Y.tail(predictions.shape[0])
else:
    outcome = Y
accuracy_matrix = outcome*predictions
accuracy_matrix.values
accuracy = (np.sum(np.sum((accuracy_matrix > 0)))/accuracy_matrix.size)

predictions_test = results.predict(exog=X_test)
predictions_test = (predictions_test.shift(-1)).dropna()
predictions_test.to_csv(os.path.dirname(__file__) + '/predicted_values_VARMAX_test.csv')

#predictions.to_csv(os.path.dirname(__file__) + '/predicted_values_VARMAX.csv')
print('Accuracy: ' + str(accuracy))
