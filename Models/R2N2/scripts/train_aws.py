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
from Models.RNN.scripts.SimpleRNN import SimpleRNN

torch.manual_seed(1)

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


#choose hyperparameters

for _ in range(50):

    n_epochs = 100
    hidden_size = np.random.randint(5,100)
    learning = np.random.randint(10,100)
    opt_num = np.random.randint(1,4)

    lr = learning * 10e-3

    #set model, loss, and optimization
    model_string = 'Simple R2N2'
    model = SimpleRNN(hidden_size= hidden_size, input_size=len(X.iloc[0:1].values[0]), output_size= len(Y.iloc[0:1].values[0]))
    model.cuda()
    criterion = nn.MSELoss()

    if opt_num == 1:
        optim_string = 'SGD'
        optimizer = optim.SGD(model.parameters(), lr=lr)
    elif opt_num == 2:
        optim_string = 'SGDM'
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    else:
        optim_string = 'Adam'
        torch.optim.Adam(model.parameters(), lr=lr)

    model_name = '{}_{}_{}_{}'.format(optim_string, hidden_size, learning, n_epochs)



    #train model
    losses = np.zeros(n_epochs) # For plotting
    time_train = time.time()

    for epoch in range(n_epochs):

        tic = time.time()

        for iter in range(len(x)):
            input = Variable(torch.from_numpy(x[iter]).float().cuda())
            target = Variable(torch.from_numpy(y[iter]).float().cuda())

            output, hidden = model.forward(input)

            optimizer.zero_grad()
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            losses[epoch] += loss.data[0]


        if epoch > 0:
            print(epoch, loss.data[0])
            print('Time of epoch: {}'.format(time.time() - tic))

    # Save losses to csv

    loss_df = pd.DataFrame({'loss': losses / len(x)})
    loss_df.to_csv(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) + '/loss_csvs/{}.csv'.format(model_name))

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

