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


#choose hyperparameters

for _ in range(50):

    n_epochs = 100
    hidden_size = np.random.randint(5,100)
    learning = np.random.randint(10,100)
    opt_num = np.random.randint(1,4)

    lr = learning * 10e-3

    #set model, loss, and optimization
    model_string = 'Simple RNN'
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
    loss_df.to_csv(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) + '/loss_csvs/{}.csv'.format(model_name))

    #Save weights
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
