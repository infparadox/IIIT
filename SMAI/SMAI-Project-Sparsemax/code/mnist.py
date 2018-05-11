from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import torch.utils.data
import random
import numpy as np
from random import randint
from sparsemax import Sparsemax
from sparsemax import MultiLabelSparseMaxLoss

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--softmax', type=bool, default=0, metavar='N',
                    help='For switching to softmax')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=2, metavar='N',
                    help='number of epochs to train (default: 2)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = args.cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.batch_size, shuffle=True, **kwargs)

def generateRandomCliqueVector(clusters, nodes_per_cluster):
	result = np.zeros(clusters*nodes_per_cluster)
	for i in range(clusters):
		j = random.randint(0,nodes_per_cluster-1)
		result[i*nodes_per_cluster+j]=1.0
	return result


class Net(nn.Module):
    def __init__(self, H_clusters, H_neurons_per_cluster):
        super(Net, self).__init__()
        self.H_clusters=H_clusters
        self.H_neurons_per_cluster=H_neurons_per_cluster
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50,self.H_clusters*self.H_neurons_per_cluster)


        self.sparsemaxActivation = Sparsemax(self.H_clusters,self.H_neurons_per_cluster)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        if args.softmax:
             return F.log_softmax(x, dim=1)
        else:
            y_pred, zs_sparse, taus, is_gt = self.sparsemaxActivation(x)
            return x, y_pred, zs_sparse, taus, is_gt

H_clusters, H_neurons_per_cluster, N_class = 1, 10, 10
model = Net(H_clusters, H_neurons_per_cluster)
sparsemaxMulticlassLoss = MultiLabelSparseMaxLoss(H_clusters, H_neurons_per_cluster)
if args.cuda:
    model.cuda()

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
code_target_class = np.zeros((N_class,H_clusters*H_neurons_per_cluster), dtype='float32')

for i in range(N_class):
    one_hot_vector = np.zeros(H_clusters*H_neurons_per_cluster)
	#code_target_class[i] = generateRandomCliqueVector(H_clusters,H_neurons_per_cluster).reshape((H_clusters*H_neurons_per_cluster))
    one_hot_vector[i] = 1.0
    code_target_class[i]=one_hot_vector

table_embedding = nn.Embedding(N_class, H_clusters*H_neurons_per_cluster, sparse=True)
table_embedding.volatile=True
table_embedding.requires_grad=False
table_embedding.weight = nn.Parameter(torch.from_numpy(code_target_class))
table_embedding.weight.requires_grad=False

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        code_target = table_embedding(target)
        optimizer.zero_grad()
        #print (model(data))
        if args.softmax:
            output = model(data)
            loss = F.nll_loss(output, target)
        else:
            input_sparsemax, y_pred, zs_sparse, taus, is_gt = model(data)
            loss = sparsemaxMulticlassLoss(input_sparsemax, zs_sparse, code_target, y_pred, taus, is_gt)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('\rTrain Epoch: {} [{}/{} ({:.3f}%)]\tLoss: {:.3f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]), end='')

def test():
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        if args.softmax:
            output = model(data)
            test_loss += F.nll_loss(output, target, size_average=False).data[0] # sum up batch loss
        else:
            data, target = Variable(data, volatile=True), Variable(target)
            _, output, _ , _ , _ = model(data)
        pred = output.data.max(1)[1]

        correct += pred.eq(target.data).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\rTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)),end = '')


for epoch in range(1, args.epochs + 1):
    train(epoch)
    print('\n')
print('\n')
test()
print('\n')
