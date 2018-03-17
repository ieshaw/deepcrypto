__author__ = 'Ian'


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#loop through models in some directory

loss_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))  + '/loss_csvs/'

output_file = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))  + '/model_compare/all_loss.csv'

loss_df = pd.DataFrame(columns=['Model', 'Opt', 'Hidden_Size', 'Learning', 'Loss'])

for file in os.listdir(loss_dir):

    model_name = file.split('.')[0]

    new_df = pd.read_csv(loss_dir + file, index_col= 0, header=0)

    new_df.plot()

    plt.title(model_name)

    plt.show()



#
#     model_list = model_name.split('_')
#
#     result_dict = {}
#
#     result_dict['Model'] = model_name
#
#     result_dict['Loss'] = new_df['loss'].loc[new_df.last_valid_index()]
#
#     result_dict['Opt'] = model_list[0]
#
#     result_dict['Hidden_Size'] = model_list[1]
#
#     result_dict['Learning'] = model_list[2]
#
#     loss_df = loss_df.append(result_dict, ignore_index= True)
#
#
# loss_df.to_csv(output_file, index=False)