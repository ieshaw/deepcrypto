__author__ = 'Ian'

#using https://gist.github.com/spro/ef26915065225df65c1187562eca7ec4

import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from statsmodels.tsa.api import VAR
from torch.autograd import Variable

from Data.scripts.data import data
from Models.R2N2.Local.scripts.RNN import RNN

torch.manual_seed(1)

def load_data():

    #import data
    X,Y = data.import_data(set= 'train')

    #do not plug in returns, but residuals
    #plug in residuals
    VAR_model = VAR(X)

    results = VAR_model.fit(1)
    ar_returns = results.fittedvalues

    #columns to drop from dataframe
    columns = ['XMRspread', 'XMRvolume', 'XMRbasevolume','XRPspread', 'XRPvolume', 'XRPbasevolume','LTCspread', 'LTCvolume', 'LTCbasevolume', 'DASHspread', 'DASHvolume', 'DASHbasevolume','ETHspread', 'ETHvolume', 'ETHbasevolume']
    ar_returns.drop(columns, 1, inplace=True)

    X = X.loc[ar_returns.index]
    x_returns = X[ar_returns.columns]
    residual_df = x_returns - ar_returns
    X = X.join(residual_df, how = 'inner', rsuffix = 'residual')

    y_ar_returns = ar_returns
    y_ar_returns.columns = Y.columns
    Y = (Y.loc[X.index] - y_ar_returns.shift(-1)).dropna()
    X = X.loc[Y.index]

    x = X.as_matrix()
    y = Y.as_matrix()

    return x,y,X,Y

x,y, X, Y = load_data()

#hyperparameters
hidden_size = 10
n_epochs = 300
learning = 10
lr = learning * 10e-3
model_string = 'Mom_LSTM_6_BFC_1_AFC_1_Act_None'
optim_string = 'SGD'

#set
input_size = len(X.iloc[0:1].values[0])
output_size = len(Y.iloc[0:1].values[0])

#set model, loss, and optimization
# model = RNN(hidden_size= hidden_size, input_size=len(X.iloc[0:1].values[0]), output_size= len(Y.iloc[0:1].values[0]))
model = RNN(hidden_size= hidden_size, input_size= input_size, output_size= output_size)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=lr, momentum= 0.01)

# add leadning rate decay

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(int(n_epochs / 3), 1), gamma=0.1)

#train model
losses = np.zeros(n_epochs) # For plotting
best_loss = np.inf
time_train = time.time()

for epoch in range(n_epochs):

    tic = time.time()

    for iter in range(len(x)):
        input = Variable(torch.from_numpy(x[iter]).float())
        target = Variable(torch.from_numpy(y[iter]).float())

        output = model.forward(input)

        optimizer.zero_grad()
        loss = criterion(output, target)
        loss.backward(retain_graph=True)
        optimizer.step()

        losses[epoch] += loss.data[0]

    scheduler.step()


    print(epoch, losses[epoch])
    print('Time of epoch: {}'.format(time.time() - tic))

    if losses[epoch] < best_loss:

        best_loss = losses[epoch]

        # Save losses to csv

        model_name = '{}_H{}'.format(model_string, hidden_size)

        loss_df = pd.DataFrame({'loss': losses[:(epoch + 1)] / len(x)})
        loss_df.to_csv(
            os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) + '/loss_csvs/{}.csv'.format(model_name))

        # Save weights
        torch.save(model.state_dict(), os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
                   + '/model_params/{}.pth.tar'.format(model_name))

        filename = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) + "/records/{}.txt".format(model_name)
        with open(filename, 'w') as f:
            f.write("{} \n".format(model_string))
            f.write("{} \n".format(model_name))
            f.write("Num_Epochs = {}\n".format(epoch + 1))
            f.write("Hidden_Size = {}\n".format(hidden_size))
            f.write("Optimizer = {}\n".format(optim_string))
            f.write("Learning Rate = {}\n".format(lr))
            f.write("Time spent = {0}\n".format(time.time() - time_train))

