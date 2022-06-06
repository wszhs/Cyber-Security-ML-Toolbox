'''
Author: your name
Date: 2021-07-27 11:01:17
LastEditTime: 2021-07-27 11:05:12
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /Cyber-Security-ML-Toolbox/csmt/classifiers/torch/lr.py
'''
'''
Author: your name
Date: 2021-03-24 21:41:48
LastEditTime: 2021-07-27 10:14:52
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
import random

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
        
class LR(nn.Module):
    def __init__(self,input_size,output_size):
        super(LR,self).__init__()
        self.linear=nn.Linear(in_features=input_size,out_features=output_size)
    def forward(self,x):
        x=self.linear(x)
        return x

class LRTorch(AbstractModel):
    """
    Multi-layer perceptron.
    """

    def __init__(self, input_size,learning_rate=0.01,
                weight_decay=0,output_size=None):
        model=LR(input_size=input_size,output_size=output_size)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.classifier = PyTorchClassifier(model=model,loss=criterion,optimizer=optimizer,input_shape=input_size,nb_classes=output_size,clip_values=(0,1))
