

# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 19:44:01 2021

@author: Robbie Sunbury
"""
from memtorch.utils import LoadCIFAR10
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F

class CIFAR10Methods:
    
    def dataReturn(self, batchSize):
        
        transform = transforms.Compose([transforms.ToTensor()])
        
        #This line corresponds to the dataset from torch
        fool_set = torchvision.datasets.CIFAR10(
            root="data", train=False, transform=transform, download=True
        )
        
        #This line corresponds to the dataset loading provided by memtorch
        train_loader, validation_loader, test_loader = LoadCIFAR10(batchSize, validation=False)
        
        return fool_set, train_loader, validation_loader, test_loader
    
    def returnNetToDevice(self, device):
        
        #These methods and values refer directly to the neural network architecture
        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.network = nn.Sequential(
                    nn.Conv2d(3, 32, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(4, 4), # output: 64 x 16 x 16
                    
        
                    nn.Flatten(), 
                    nn.Linear(256*4*4, 512),
                    nn.ReLU(),
                    nn.Linear(512, 10))
        
            def forward(self, x):
                return self.network(x)
            
        net = Net()
        return net.to(device)
    
    def getName(self):
        return "CIFAR10_model.pt"