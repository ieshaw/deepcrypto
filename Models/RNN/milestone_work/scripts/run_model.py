__author__ = 'Ian'

#using https://gist.github.com/spro/ef26915065225df65c1187562eca7ec4

import numpy as np
import os
import pandas as pd
import time
import torch
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math, random

from Data.scripts.data import data
from Models.RNN.SimpleRNN import SimpleRNN


torch.manual_seed(1)

#import data
X,Y = data.import_data(set= 'train')
x = X.as_matrix()
y = Y.as_matrix()

#set preditcion matrix
y_pred = np.zeros(shape= y.shape)

#set model
model = SimpleRNN(hidden_size= 10, input_size=len(X.iloc[0:1].values[0]), output_size= len(Y.iloc[0:1].values[0]))
model.load_state_dict(torch.load(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) + '/model_parmas.pth.tar'))

tic = time.time()

for iter in range(len(x)):
    input = Variable(torch.from_numpy(x[iter]).float())
    target = Variable(torch.from_numpy(y[iter]).float())

    output, hidden = model.forward(input)

    y_pred[iter] = output.data.numpy()

Y_pred = pd.DataFrame(data= y_pred, index= Y.index, columns= Y.columns)
# Y_pred = (Y_pred.shift(1)).dropna()

Y_pred.to_csv(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) + '/csvs/y_pred.csv')

print('Time of evaluation: {}'.format(time.time() - tic))