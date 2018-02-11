__author__ = 'Ian'

#using https://gist.github.com/spro/ef26915065225df65c1187562eca7ec4

import numpy as np
import torch
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math, random
import time

from Data.scripts.data import data

torch.manual_seed(1)

class SimpleRNN(nn.Module):
    def __init__(self, hidden_size, input_size, output_size):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size

        self.inp = nn.Linear(self.input_size, hidden_size)
        self.rnn = nn.LSTM(hidden_size, hidden_size, 2, dropout=0.05)
        self.out = nn.Linear(hidden_size, self.output_size)

    def forward(self, input, hidden=None):
        input = self.inp(input.view(1, -1)).unsqueeze(1)
        output, hidden = self.rnn(input, hidden)
        output = self.out(output.squeeze(1))
        return output, hidden

#import data
X,Y = data.import_data(set= 'train')
#shorten dataset for development
x = X.as_matrix()
y = Y.as_matrix()
#run and train model

n_epochs = 100
n_iters = 50

model = SimpleRNN(hidden_size= 10, input_size=len(X.iloc[0:1].values[0]), output_size= len(Y.iloc[0:1].values[0]))
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

losses = np.zeros(n_epochs) # For plotting

#print inital prediciton



for epoch in range(n_epochs):

    tic = time.time()

    for iter in range(len(x)-1):
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


