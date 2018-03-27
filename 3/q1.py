#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 25 18:04:30 2018

@author: Smiker
"""

import torch
import torch.utils.data
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.serialization import load_lua
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

data = load_lua('dataset.t7b')

X = data.data
y = data.label
size = X.size()[0]
shuffle = torch.randperm(size)
X = X[shuffle]
y = y[shuffle]

tr_size = (int) (size * 0.9)

Xtr = X[:tr_size,:]
ytr = y[:tr_size]
Xts = X[tr_size:,:]
yts = y[tr_size:]
ytr = ytr - 1
yts = yts - 1

Xtr = Variable(Xtr)
ytr = Variable(ytr.long(), requires_grad = False)
Xts = Variable(Xts)
yts = Variable(yts.long(), requires_grad = False)

class Net(nn.Module):
    def __init__(self, H):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(2, H)
        self.fc2 = nn.Linear(H, 4)

    def forward(self, x):
        x = F.sigmoid(self.fc1(x))
        x = self.fc2(x)
        return x

H = 10

net = Net(H)
m = nn.LogSoftmax()

criterion = nn.NLLLoss()
optimizer = optim.SGD(net.parameters(), lr=5)
loss_trs = []
loss_tss = []
T = np.arange(0,1000,1)
for t in T:
    y_pred = net(Xtr)
    yts_pred = net(Xts)

    loss_tr = criterion(m(y_pred), ytr)
    loss_trs.append(loss_tr.data[0])
    
    loss_ts = criterion(m(yts_pred), yts)
    loss_tss.append(loss_ts.data[0])
    
    optimizer.zero_grad()
    loss_tr.backward()
    optimizer.step()
    

plt.plot(T, loss_trs, 'r')
plt.plot(T, loss_tss, 'b')
plt.ylim(0.18,0.4)
plt.legend(['Train loss','Test loss'], loc='upper right')

_, tr_predicted = torch.max(y_pred.data, 1)
_, ts_predicted = torch.max(yts_pred.data, 1)
tr_correct = (tr_predicted == ytr.data).sum()
ts_correct = (ts_predicted == yts.data).sum()
tr_arc = tr_correct / tr_size
ts_arc = ts_correct / (size - tr_size)
print(tr_arc)
print(ts_arc)
