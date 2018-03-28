
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim
import numpy


# In[2]:


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
    return (tr1 == tr2).type(torch.LongTensor)

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]
        
def get_even_data(data, labels, count):
    true_idx = labels.numpy().nonzero()
    false_idx = (labels == 0).numpy().nonzero()
    #print(data[false_idx].size())
    #print(data[true_idx].size())
    new_data = torch.cat((data[true_idx][:count], data[false_idx][:count]))
    new_labels = torch.cat((labels[true_idx][:count], labels[false_idx][:count]))
    return new_data, new_labels


# In[3]:


## load mnist dataset
use_cuda = torch.cuda.is_available()

root = './data'
download = False  # download MNIST dataset or not

trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
train_set = dset.MNIST(root=root, train=True, transform=trans, download=download)
test_set = dset.MNIST(root=root, train=False, transform=trans)

tr_data = split_and_stack(train_set.train_data)
ts_data = split_and_stack(test_set.test_data)
tr_labels = get_labels(train_set.train_labels)
ts_labels = get_labels(test_set.test_labels)

train_data, train_labels = get_even_data(tr_data, tr_labels, 2000)
test_data, test_labels = get_even_data(ts_data, ts_labels, 500)

batch_size = 500

# train_loader = torch.utils.data.DataLoader(
#                  dataset=train_set,
#                  batch_size=batch_size,
#                  shuffle=True)
# test_loader = torch.utils.data.DataLoader(
#                 dataset=test_set,
#                 batch_size=batch_size,
#                 shuffle=False)

ntrain = len(train_data) // batch_size
ntest = len(test_data) // batch_size

print('==>>> total trainning batch number: {}'.format(ntrain))
print('==>>> total testing batch number: {}'.format(ntest))


# In[4]:



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


# In[9]:


## training
model = LeNet()

if use_cuda:
    model = model.cuda()

optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)

ceriation = nn.CrossEntropyLoss()


# In[10]:


for epoch in range(50):
    train_data_iter = chunks(train_data, batch_size)
    train_labels_iter = chunks(train_labels, batch_size)
    test_data_iter = chunks(test_data, batch_size)
    test_labels_iter = chunks(test_labels, batch_size)
    # trainning
    ave_loss = 0
    for batch_idx, (x, target) in enumerate(zip(train_data_iter, train_labels_iter)):
        optimizer.zero_grad()
        if use_cuda:
            x, target = x.cuda(), target.cuda()
        x, target = Variable(x), Variable(target)
        out = model(x)
        loss = ceriation(out, target)
        ave_loss = ave_loss * 0.9 + loss.data[0] * 0.1
        loss.backward()
        optimizer.step()
        if (batch_idx+1) % 100 == 0 or (batch_idx+1) == ntrain:
            print('==>>> epoch: {}, batch index: {}, train loss: {:.6f}'.format(
                epoch, batch_idx+1, ave_loss))
    # testing
    correct_cnt, ave_loss = 0, 0
    total_cnt = 0
    for batch_idx, (x, target) in enumerate(zip(test_data_iter, test_labels_iter)):
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
        
        if(batch_idx+1) % 100 == 0 or (batch_idx+1) == ntest:
            print('==>>> epoch: {}, batch index: {}, test loss: {:.6f}, acc: {:.3f}'.format(
                epoch, batch_idx+1, ave_loss, correct_cnt * 1.0 / total_cnt))

#torch.save(model.state_dict(), model.name())


# In[7]:


ave_loss

