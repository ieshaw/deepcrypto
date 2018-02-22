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

#set model, loss, and optimization
model = SimpleRNN(hidden_size= 10, input_size=len(X.iloc[0:1].values[0]), output_size= len(Y.iloc[0:1].values[0]))
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

#train model
n_epochs = 100
losses = np.zeros(n_epochs) # For plotting

for epoch in range(n_epochs):

    tic = time.time()

    for iter in range(len(x)):
        input = Variable(torch.from_numpy(x[iter]).float())
        target = Variable(torch.from_numpy(y[iter]).float())

        output, hidden = model.forward(input)

        optimizer.zero_grad()
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        losses[epoch] += loss.data[0]


    if epoch > 0:
        print(epoch, loss.data[0])
        print('Time of epoch: {}'.format(time.time() - tic))

    # Use some plotting library
    # if epoch % 10 == 0:
        # show_plot('inputs', _inputs, True)
        # show_plot('outputs', outputs.data.view(-1), True)
        # show_plot('losses', losses[:epoch] / n_iters)

        # Generate a test
        # outputs, hidden = model(inputs, False, 50)
        # show_plot('generated', outputs.data.view(-1), True)

#Save losses to csv

loss_df = pd.DataFrame({'loss':losses/len(x)})
loss_df.to_csv(os.path.dirname(__file__) + '/milestone_work/csvs/loss.csv')

#Save weights
torch.save(model.state_dict(), os.path.dirname(__file__) + '/milestone_work/model_parmas.pth.tar')

