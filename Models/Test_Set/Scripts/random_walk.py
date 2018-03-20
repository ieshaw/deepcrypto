__author__ = 'Ian'

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from Data.scripts.data import data
from sklearn.metrics import mean_squared_error, roc_auc_score, roc_curve, auc
from Models.Evaluation.eval import eval_model


_, Y_actual = data.import_data(set='test')

y_pred_df = Y_actual.shift(1)

y_pred_df = y_pred_df.dropna()

Y_actual = Y_actual.loc[y_pred_df.index]

tester = eval_model(y_pred_df= y_pred_df, y_actual_df= Y_actual)

tester.backtest(printer=False)

out_dict = tester.metrics

out_dict['mse'] = mean_squared_error(Y_actual.as_matrix(), y_pred_df.as_matrix())

flat_pred = np.clip(y_pred_df.as_matrix().flatten() + 0.5, 0, 1)

flat_actual = np.where(Y_actual.as_matrix().flatten() > 0, 1, 0)

out_dict['auc'] = roc_auc_score(flat_actual, flat_pred)
model = 'Test'
print(out_dict)

fpr, tpr, _ = roc_curve(flat_actual, flat_pred)
roc_auc = auc(fpr, tpr)

plt.figure()

lw = 2

plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('{} Receiver operating characteristic Curve'.format(model))
plt.legend(loc="lower right")

plt.show()
