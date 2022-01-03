#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 10:48:30 2020

@author: liuwenliang
"""

import torch
import numpy as np

from torch import nn
import torch.utils.data as Data
import datetime
import torch.nn.functional as F



# torch.manual_seed(1)    # reproducible

# Hyper Parameters
EPOCH = 300          
BATCH_SIZE = 128
TIME_STEP = 10          
INPUT_SIZE = 3         
HIDDEN_SIZE = 64
NUM_LAYERS = 2
LR = 0.01               # learning rate

U = torch.from_numpy(np.load('U_case1.npy')[:500,:,:]).float()
Q = torch.from_numpy(np.load('Q_case1.npy')[:500,:,:20]).float()
U_t = torch.from_numpy(np.load('U_case1.npy')[500:,:,:]).float()
Q_t = torch.from_numpy(np.load('Q_case1.npy')[500:,:,:20]).float()

U = torch.transpose(U,1,2)
Q = torch.transpose(Q,1,2)
U_t = torch.transpose(U_t,1,2)
Q_t = torch.transpose(Q_t,1,2)

Dataset = Data.TensorDataset(Q, U)


loader = Data.DataLoader(
    dataset=Dataset,      # torch TensorDataset format
    batch_size=BATCH_SIZE,      # mini batch size
    shuffle=True,               
    num_workers=2,              
)




def tansig(x,u_min,u_max):
    return u_min + (u_max-u_min)*(F.tanh(x)+1)/2

class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()

        self.rnn = nn.LSTM(       
            input_size=INPUT_SIZE,
            hidden_size=HIDDEN_SIZE,         # rnn hidden unit
            num_layers=NUM_LAYERS,           # number of rnn layer
            batch_first=True,       # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
            dropout = 0.5,
#            nonlinearity = 'relu'
        )
        
        self.out = nn.Linear(HIDDEN_SIZE, 2)

    def forward(self, x, h_state):    
        # x (batch, time_step, input_size)
        # r_out (batch, time_step, hidden_size)
        # h_n shape (n_layers, batch, hidden_size)
        # h_c shape (n_layers, batch, hidden_size)
        r_out, h_state = self.rnn(x, h_state)

        outs = []    # save all predictions
        for time_step in range(r_out.size(1)):    # calculate output for each time step
            outs.append(tansig(self.out(r_out[:, time_step, :]),torch.tensor([0.,-3.]),torch.tensor([2.,3.])))
        return torch.stack(outs, dim=1), h_state

rnn = RNN()
print(rnn)
print(rnn.parameters())

optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)  
loss_func = nn.MSELoss()                       
loss_best = 100
print(optimizer.param_groups[0]['lr'])
start = datetime.datetime.now()
# training and testing
for epoch in range(EPOCH):
    for step, (b_x, b_y) in enumerate(loader):        # gives batch data

        output, h_state = rnn(b_x,None)                               # rnn output
        loss = loss_func(output, b_y)                 
        optimizer.zero_grad()                           # clear gradients for this training step
        loss.backward()                                 # backpropagation, compute gradients
        optimizer.step()                                # apply gradients
        
        rnn.eval()
        test_output, h_state_t = rnn(Q_t,None)                  
        loss_t = loss_func(test_output, U_t)
        rnn.train()
        if step % 5 == 0:
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy(), '| test loss: %.4f' % loss_t.data.numpy())
        if loss_t.data.numpy() < loss_best:
            loss_best = loss_t.data.numpy()
            # torch.save(rnn, 'rnn_case1.pkl')
            print('test loss: %.4f' % loss_t.data.numpy(), 'Save!')
end = datetime.datetime.now()
print('Training time:')
print(end-start)
