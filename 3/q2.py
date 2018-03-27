#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 23 20:20:11 2018

@author: Smiker
"""

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import PolynomialFeatures



X = 20*(torch.rand(5000,2)-0.5)
#X = torch.rand(5000,2)
x = X[:,0]
y = X[:,1]
#x = torch.linspace(-10,10,5000)
#y = torch.linspace(-10,10,5000)
f = x ** 2 + x * y + y ** 2
#X = torch.stack((x,y),1)

size = x.size()[0]
tr_size = (int) (size * 0.9)

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(2, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

H = 10

#poly = PolynomialFeatures(2)
#x = poly.fit_transform(X.numpy())

#x = Variable(torch.from_numpy(x))
x = Variable(X)
y = Variable(f, requires_grad=False)

Xtr = x[:tr_size,:]
ytr = y[:tr_size]
Xts = x[tr_size:,:]
yts = y[tr_size:]

net = Net()

criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.0001)
T = np.arange(0,5000,1)

loss_trs = []
loss_tss = []

for t in T:
    y_pred = net(Xtr)
    yts_pred = net(Xts)
    
    loss_tr = criterion(y_pred, ytr)
    loss_trs.append(loss_tr.data[0])
    
    loss_ts = criterion(yts_pred, yts)
    loss_tss.append(loss_ts.data[0])
    #print(t, loss.data[0])

    optimizer.zero_grad()
    loss_tr.backward()
    optimizer.step()
    
    
plt.plot(T, loss_trs, 'r')
plt.plot(T, loss_tss, 'b')
#plt.ylim(0.18,0.4)
plt.legend(['Train loss','Test loss'], loc='upper right')
