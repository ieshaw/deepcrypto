__author__ = 'Ian'

import matplotlib.pyplot as plt
import os
import pandas as pd


model_string = 'VAR_EXTRA_LSTM_6_BFC_1_AFC_1_Act_None'
hidden_size = 10
model_name = '{}_H{}'.format(model_string, hidden_size)

loss_df = pd.read_csv(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) + '/loss_csvs/{}.csv'
                      .format(model_name), index_col = 0)

loss_df[1:].plot()
plt.show()