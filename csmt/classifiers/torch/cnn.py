'''
Author: your name
Date: 2021-07-12 09:43:34
LastEditTime: 2021-07-28 21:17:17
LastEditors: your name
Description: In User Settings Edit
FilePath: /Cyber-Security-ML-Toolbox/csmt/classifiers/torch/cnn.py
'''
'''
Author: your name
Date: 2021-07-12 09:43:34
LastEditTime: 2021-07-27 10:52:29
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /Cyber-Security-ML-Toolbox/csmt/classifiers/torch/cnn.py
'''
'''
Author: your name
Date: 2021-03-24 21:41:48
LastEditTime: 2021-07-12 09:43:16
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /NashHE/nashhe/ml_ids/torch/multi_layer_perceptron.py
'''
from collections import OrderedDict
from csmt.classifiers.abstract_model import AbstractModel
import torch
import torch.nn as nn
import numpy as np
from csmt.estimators.classification.pytorch import PyTorchClassifier
import torch.nn.functional as F

class LeNet_Mnist(torch.nn.Module):
    """Network architecture from: https://github.com/ChawDoe/LeNet5-MNIST-PyTorch."""
    def __init__(self,input_size,output_size):
        super().__init__()
        self.conv_1 = torch.nn.Conv2d(1, 6, 5)
        self.pool_1 = torch.nn.MaxPool2d(2, 2)
        self.relu_1 = torch.nn.ReLU()
        self.conv_2 = torch.nn.Conv2d(6, 16, 5)
        self.pool_2 = torch.nn.MaxPool2d(2, 2)
        self.relu_2 = torch.nn.ReLU()
        self.fc_1 = torch.nn.Linear(256, 120)
        self.relu_3 = torch.nn.ReLU()
        self.fc_2 = torch.nn.Linear(120, 84)
        self.relu_4 = torch.nn.ReLU()
        self.fc_3 = torch.nn.Linear(84, output_size)

    def forward(self, x):
        x = self.pool_1(self.relu_1(self.conv_1(x)))
        x = self.pool_2(self.relu_2(self.conv_2(x)))
        x = x.view(x.shape[0], -1)
        x = self.relu_3(self.fc_1(x))
        x = self.relu_4(self.fc_2(x))
        x = self.fc_3(x)
        return x

class CNN_Mnist(nn.Module):
    def __init__(self,input_size,output_size):
        super(CNN_Mnist, self).__init__()
        self.conv_1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=5, stride=1)
        self.conv_2 = nn.Conv2d(in_channels=4, out_channels=10, kernel_size=5, stride=1)
        self.fc_1 = nn.Linear(in_features= 4* 4 * 10, out_features=100)
        self.fc_2 = nn.Linear(in_features=100, out_features=output_size)

    def forward(self, x):
        x = F.relu(self.conv_1(x))
        x = F.max_pool2d(x,2,2)
        x = F.relu(self.conv_2(x))
        x = F.max_pool2d(x,2,2)
        x = x.view(-1, 4 * 4 * 10)
        x = F.relu(self.fc_1(x))
        x = self.fc_2(x)
        return x

class CNN(nn.Module):
    def __init__(self,input_size,output_size):
        super(CNN, self).__init__()
        self.conv_1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=1, stride=1)
        self.fc_1 = nn.Linear(in_features= 4* 1 * 67, out_features=10)
        self.fc_2 = nn.Linear(in_features=10, out_features=output_size)

    def forward(self, x):
        print(x.shape)
        x=x.view(-1,1,1,67)
        x = F.relu(self.conv_1(x))
        # print(x.shape)
        x = x.view(-1, 4 * 1 * 67)
        x = F.relu(self.fc_1(x))
        x = self.fc_2(x)
        return x




class CNNTorch(AbstractModel):
    """
    Multi-layer perceptron.
    """

    def __init__(self, input_size,learning_rate=0.01,
                weight_decay=0,output_size=None):
        model=CNN(input_size=input_size,output_size=output_size)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.classifier = PyTorchClassifier(model=model,loss=criterion,optimizer=optimizer,input_shape=input_size,nb_classes=output_size,clip_values=(0,1))

class CNNMnistTorch(AbstractModel):

    def __init__(self, input_size,learning_rate=0.01,
                weight_decay=0,output_size=None):
        model=CNN_Mnist(input_size=input_size,output_size=output_size)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.classifier = PyTorchClassifier(model=model,loss=criterion,optimizer=optimizer,input_shape=input_size,nb_classes=output_size,clip_values=(0,1))

class LeNetMnistTorch(AbstractModel):

    def __init__(self, input_size,learning_rate=0.01,
                weight_decay=0,output_size=None):
        model=LeNet_Mnist(input_size=input_size,output_size=output_size)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.classifier = PyTorchClassifier(model=model,loss=criterion,optimizer=optimizer,input_shape=input_size,nb_classes=output_size,clip_values=(0,1))
