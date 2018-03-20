__author__ = 'Ian'

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from Data.scripts.data import data
from sklearn.metrics import mean_squared_error, roc_auc_score, roc_curve, auc
from Models.Evaluation.eval import eval_model


_, Y_actual = data.import_data(set='test')

# out_df = pd.DataFrame(columns=['return', 'sharpe', 'max_drawdown', 'auc', 'mse'])

OUT = {}

pred_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) + '/Pred_CSVs/'

for file in os.listdir(pred_dir):

    model = file.split('_')[0]

    file = pred_dir + file

    y_pred_df = pd.read_csv(file, index_col= 0)

    tester = eval_model(y_pred_df= y_pred_df, y_actual_df= Y_actual)

    tester.backtest(printer=False)

    strat_out_file = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) + '/Out_CSVs/{}.csv'.format(model)

    out_dict = tester.metrics

    tester.strat_series.to_csv(strat_out_file, header = True)

    print(model)

    out_dict['mse'] = mean_squared_error(Y_actual.as_matrix(), y_pred_df.as_matrix())

    flat_pred = np.clip(y_pred_df.as_matrix().flatten() + 0.5, 0, 1)

    flat_actual = np.where(Y_actual.as_matrix().flatten() > 0, 1, 0)

    out_dict['auc'] = roc_auc_score(flat_actual, flat_pred)

    OUT[model] = out_dict

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
    plt.savefig(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) + '/Plots/{}_auc.png'.format(model))


out_df = pd.DataFrame.from_dict(OUT, orient = 'index')

out_df.to_csv(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) + '/results.csv')

out_df.to_latex(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) + '/results_table.tex')
