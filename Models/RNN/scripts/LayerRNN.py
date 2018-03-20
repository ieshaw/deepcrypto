__author__ = 'Ian'

#using https://gist.github.com/spro/ef26915065225df65c1187562eca7ec4


import torch
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class LayerRNN(nn.Module):
    def __init__(self, hidden_size, input_size, output_size):
        super(LayerRNN, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size
        self.inp = nn.Linear(self.input_size, hidden_size)
        self.rnn = nn.LSTM(hidden_size, hidden_size, 2, dropout=0.05)
        self.out = nn.Linear(hidden_size, self.output_size)
        #self.out2 = nn.Linear(self.output_size, self.output_size)
        # self.out3 = nn.Linear(self.output_size, self.output_size)
        # self.relu = nn.ReLU()
        #self.lekrel= nn.LeakyReLU(0.1)
        self.hidden = self.init_hidden()

    def init_hidden(self, x=None):
        if x == None:
            return (Variable(torch.zeros(2, 1, self.hidden_size)),
                    Variable(torch.zeros(2, 1, self.hidden_size)))
        else:
            return (Variable(x[0].data), Variable(x[1].data))

    def forward(self, input):
        input = self.inp(input.view(1, -1)).unsqueeze(1)
        output, self.hidden_out = self.rnn(input, self.hidden)
        output = self.out(output.squeeze(1))
        #output = self.relu(output)
        #output = self.lekrel(output)
        #output = self.out2(output)
        # output = self.relu(output)
        # output = self.out3(output)
        self.hidden = self.init_hidden(self.hidden_out)
        return output

