
# coding: utf-8

# In[55]:


import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim


# In[66]:


def split_and_stack(data):
    n = data.size()[0]
    half = n // 2
    tr1 = data[:half]
    tr2 = data[half:]
    tr_data = torch.stack((tr1, tr2), 1)
    return tr_data.type(torch.FloatTensor)

def get_labels(labels):
    n = labels.size()[0]
    half = n // 2
    tr1 = labels[:half]
    tr2 = labels[half:]
    return (tr1 == tr2).type(torch.IntTensor)


# In[67]:


## load mnist dataset
use_cuda = torch.cuda.is_available()

root = './data'
download = False  # download MNIST dataset or not

trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
train_set = dset.MNIST(root=root, train=True, transform=trans, download=download)
test_set = dset.MNIST(root=root, train=False, transform=trans)

train_data = split_and_stack(train_set.train_data)
test_data = split_and_stack(test_set.test_data)
train_labels = get_labels(train_set.train_labels)
test_labels = get_labels(test_set.test_labels)
print(train_labels.shape)

batch_size = 1

# train_loader = torch.utils.data.DataLoader(
#                  dataset=train_set,
#                  batch_size=batch_size,
#                  shuffle=True)
# test_loader = torch.utils.data.DataLoader(
#                 dataset=test_set,
#                 batch_size=batch_size,
#                 shuffle=False)

print('==>>> total trainning batch number: {}'.format(len(train_data)))
print('==>>> total testing batch number: {}'.format(len(test_data)))


# In[68]:



## network
class MLPNet(nn.Module):
    def __init__(self):
        super(MLPNet, self).__init__()
        self.fc1 = nn.Linear(28*28, 500)
        self.fc2 = nn.Linear(500, 256)
        self.fc3 = nn.Linear(256, 10)
    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def name(self):
        return "MLP"

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        ##modified input channel and output channel
        self.conv1 = nn.Conv2d(2, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    def name(self):
        return "LeNet"


# In[69]:


## training
model = LeNet()

if use_cuda:
    model = model.cuda()

optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

ceriation = nn.CrossEntropyLoss()

train_data, train_labels = Variable(train_data), Variable(train_labels)



# In[71]:


for epoch in range(1):
    # trainning
    ave_loss = 0
    for batch_idx in range(1):
        x = train_data
        target = train_labels
        optimizer.zero_grad()
        if use_cuda:
            x, target = x.cuda(), target.cuda()
        ##x, target = Variable(x), Variable(target)
        out = model(x)
        loss = ceriation(out, target)
        ave_loss = ave_loss * 0.9 + loss.data[0] * 0.1
        loss.backward()
        optimizer.step()
        if (batch_idx+1) % 100 == 0 or (batch_idx+1) == len(train_data):
            print('==>>> epoch: {}, batch index: {}, train loss: {:.6f}'.format(
                epoch, batch_idx+1, ave_loss))
    # testing
    correct_cnt, ave_loss = 0, 0
    total_cnt = 0
    for batch_idx, (x, target) in (test_data, test_labels):
        if use_cuda:
            x, targe = x.cuda(), target.cuda()
        x, target = Variable(x, volatile=True), Variable(target, volatile=True)
        out = model(x)
        loss = ceriation(out, target)
        _, pred_label = torch.max(out.data, 1)
        total_cnt += x.data.size()[0]
        correct_cnt += (pred_label == target.data).sum()
        # smooth average
        ave_loss = ave_loss * 0.9 + loss.data[0] * 0.1
        
        if(batch_idx+1) % 100 == 0 or (batch_idx+1) == len(test_data):
            print('==>>> epoch: {}, batch index: {}, test loss: {:.6f}, acc: {:.3f}'.format(
                epoch, batch_idx+1, ave_loss, correct_cnt * 1.0 / total_cnt))

torch.save(model.state_dict(), model.name())

