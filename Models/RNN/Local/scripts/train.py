__author__ = 'Ian'

#using https://gist.github.com/spro/ef26915065225df65c1187562eca7ec4

import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from Data.scripts.data import data
from Models.RNN.scripts.SimpleRNN import SimpleRNN

torch.manual_seed(1)

#import data
X,Y = data.import_data(set= 'train')
x = X.as_matrix()
y = Y.as_matrix()

#set model, loss, and optimization
hidden_size = 10
optim_string = 'SGD'
model_string = 'Simple RNN'
n_epochs = 200
learning = 100
lr = learning * 10e-3
model = SimpleRNN(hidden_size= hidden_size, input_size=len(X.iloc[0:1].values[0]), output_size= len(Y.iloc[0:1].values[0]))
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr= lr)

# add leadning rate decay

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(int(n_epochs / 3), 1), gamma=0.1)


#train model
losses = np.zeros(n_epochs) # For plotting

time_train = time.time()

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
        scheduler.step()

        losses[epoch] += loss.data[0]


    if epoch > 0:
        print(epoch, loss.data[0])
        print('Time of epoch: {}'.format(time.time() - tic))

    if epoch + 1 % 100:

        # Save losses to csv

        model_name = '{}_{}_{}_{}'.format(optim_string, hidden_size, learning, epoch + 1)

        loss_df = pd.DataFrame({'loss': losses[:epoch] / len(x)})
        loss_df.to_csv(
            os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) + '/loss_csvs/{}.csv'.format(model_name))

        # Save weights
        torch.save(model.state_dict(), os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
                   + '/model_params/{}.pth.tar'.format(model_name))

        filename = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) + "/records/{}.txt".format(model_name)
        with open(filename, 'w') as f:
            f.write("{} \n".format(model_string))
            f.write("{} \n".format(model_name))
            f.write("Num_Epochs = {}\n".format(n_epochs))
            f.write("Hidden_Size = {}\n".format(hidden_size))
            f.write("Optimizer = {}\n".format(optim_string))
            f.write("Learning Rate = {}\n".format(lr))
            f.write("Time spent = {0}\n".format(time.time() - time_train))




