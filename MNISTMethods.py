# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 19:44:01 2021

@author: Robbie Sunbury
"""
from memtorch.utils import LoadMNIST
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F

class MNISTMethods:
    
    def dataReturn(self, batchSize):
        
        transform = transforms.Compose([transforms.ToTensor()])
        fool_set = torchvision.datasets.MNIST(
            root="data", train=False, transform=transform, download=True
        )
        
        train_loader, validation_loader, test_loader = LoadMNIST(batchSize, validation=False)
        
        return fool_set, train_loader, validation_loader, test_loader
    
    def returnNetToDevice(self, device):
        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.conv1 = nn.Conv2d(1, 20, 5, 1)
                self.conv2 = nn.Conv2d(20, 50, 5, 1)
                self.fc1 = nn.Linear(4*4*50, 500)
                self.fc2 = nn.Linear(500, 10)
        
            def forward(self, x):
                x = F.relu(self.conv1(x))
                x = F.max_pool2d(x, 2, 2)
                x = F.relu(self.conv2(x))
                x = F.max_pool2d(x, 2, 2)
                x = x.view(-1, 4*4*50)
                x = F.relu(self.fc1(x))
                x = self.fc2(x)
                return x
            
        return Net().to(device)
    
    def getName(self):
        return "MNIST_model.pt"