__author__ = 'Ian'

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

strat_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) + '/Out_CSVs/'

i = 0

for file in os.listdir(strat_dir):

    new_df = pd.read_csv(strat_dir + file, index_col= 0)

    model = file.split('.')[0]

    new_df.columns = [model]

    new_df = new_df.reset_index()

    if i ==0:

        comp_df = new_df

    else:

        # comp_df = pd.concat([comp_df, new_df], ignore_index= True)
        comp_df[model] = new_df[model]

    i +=1

comp_df.plot()
plt.title('Test Set Backtest')
plt.ylabel('Return')
plt.xlabel('Hours')
plt.savefig(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) + '/Plots/backtest_novar.png')
plt.show()

