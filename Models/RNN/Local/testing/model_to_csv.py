__author__ = 'Ian'

import os
import time

import numpy as np
import pandas as pd
import torch
from torch.autograd import Variable

from Data.scripts.data import data
from Models.RNN.scripts.SimpleRNN import SimpleRNN
from Models.RNN.scripts.LayerRNN import LayerRNN
from Models.Evaluation.eval import eval_model

X, Y = data.import_data(set='cross_val')
x = X.as_matrix()
y = Y.as_matrix()

model_string = 'RNN'

model_params_file_str = '/Users/ianshaw/Downloads/GitHub/deepcrypto/Models/RNN/Local/model_params/Layer1_Relu_hiddenfor_H10.pth.tar'

hidden_size = 10

model = LayerRNN(hidden_size=hidden_size, input_size=len(X.iloc[0:1].values[0]),
                 output_size=len(Y.iloc[0:1].values[0]))
model.load_state_dict(torch.load(model_params_file_str))

#run it through the cross val to develop memory

for iter in range(len(x)):
    input = Variable(torch.from_numpy(x[iter]).float())
    target = Variable(torch.from_numpy(y[iter]).float())

    output = model.forward(input)


#run it through the test to output csv

X, Y = data.import_data(set='test')
x = X.as_matrix()
y = Y.as_matrix()

# set preditcion matrix
y_pred = np.zeros(shape=y.shape)

for iter in range(len(x)):
    input = Variable(torch.from_numpy(x[iter]).float())
    target = Variable(torch.from_numpy(y[iter]).float())

    output = model.forward(input)

    y_pred[iter] = output.data.numpy()

Y_pred = pd.DataFrame(data=y_pred, index=Y.index, columns=Y.columns)

Y_pred.to_csv('{}_test.csv'.format(model_string))