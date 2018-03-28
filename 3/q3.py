#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 18:52:08 2018

@author: Smiker
"""

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision      # 数据库模块
import matplotlib.pyplot as plt
import torch.nn.functional as F


torch.manual_seed(1)    # reproducible

# Hyper Parameters
EPOCH = 1           # 训练整批数据多少次, 为了节约时间, 我们只训练一次
BATCH_SIZE = 50
LR = 0.001          # 学习率
DOWNLOAD_MNIST = True  # 如果你已经下载好了mnist数据就写上 Fasle


# Mnist 手写数字
train_data = torchvision.datasets.MNIST(
    root='./mnist/',    # 保存或者提取位置
    train=True,  # this is training data
    transform=torchvision.transforms.ToTensor(),    # 转换 PIL.Image or numpy.ndarray 成
                                                    # torch.FloatTensor (C x H x W), 训练的时候 normalize 成 [0.0, 1.0] 区间
    download=DOWNLOAD_MNIST,
)

test_data = torchvision.datasets.MNIST(root='./mnist/', train=False)

# 批训练 50samples, 1 channel, 28x28 (50, 1, 28, 28)
train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

train_size = (int)(0.1 * train_data.train_data.size()[0])
test_size = (int)(0.1 * test_data.test_data.size()[0])
train_datas = train_data.train_data[:train_size]
train_label = train_data.train_labels[:train_size]
test_datas = test_data.test_data[:test_size]
test_label = test_data.test_labels[:test_size]
#train_datas = train_datas[:,None,:,:]
#test_datas = test_datas[:,None,:,:]

#Modify training data
tr1 = train_datas[:3000]
tr2 = train_datas[3000:]
tr_data = torch.stack((tr1, tr2), 1)
tr_label1 = train_label[:3000]
tr_label2 = train_label[3000:]
tr_label = (tr_label1 == tr_label2)


ts1 = test_datas[:500]
ts2 = test_datas[500:]
ts_data = torch.stack((ts1, ts2), 1)
ts_label1 = test_label[:500]
ts_label2 = test_label[500:]
ts_label = (ts_label1 == ts_label2)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(2, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.sigmoid(self.fc3(x))
        return x

cnn = CNN()

optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)   # optimize all cnn parameters
loss_func = nn.CrossEntropyLoss()   # the target label is not one-hotted

# training and testing
for epoch in range(EPOCH):
    b_x = Variable(tr_data)   # batch x
    b_y = Variable(tr_label)   # batch y

    output = cnn(b_x)               # cnn output
    loss = loss_func(output, b_y)   # cross entropy loss
    optimizer.zero_grad()           # clear gradients for this training step
    loss.backward()                 # backpropagation, compute gradients
    optimizer.step()                # apply gradients
        
#test_output = cnn(test_x[:10])
#pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
#print(pred_y, 'prediction number')
#print(test_y[:10].numpy(), 'real number')