__author__ = 'Ian'

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from Data.scripts.data import data

pred_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) + '/Pred_CSVs/'

_, Y_actual = data.import_data(set='test')

y_mat = Y_actual.as_matrix()

for file in os.listdir(pred_dir):

    new_df = pd.read_csv(pred_dir + file, index_col= 0)

    model = file.split('_')[0]

    new_mat = new_df.as_matrix()

    flat_new = new_mat.flatten()

    diff_mat = y_mat - new_mat

    flat_diff = diff_mat.flatten()

    plt.hist(flat_new, bins= 100)
    plt.title('{} Pred Hist'.format(model))
    plt.savefig(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) + '/extra_plots/{}_pred_hist.png'.format(model))
    plt.show()
    plt.clf()

