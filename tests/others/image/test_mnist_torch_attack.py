'''
Author: your name
Date: 2021-06-24 15:56:14
LastEditTime: 2021-07-27 14:28:32
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /Cyber-Security-ML-Toolbox/tests/image/test_mnist_attack.py
'''
'''
Author: your name
Date: 2021-06-24 14:35:20
LastEditTime: 2021-06-24 15:40:28
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /Cyber-Security-ML-Toolbox/tests/image/test_mnist_torch.py
'''
import sys
from numpy.core.defchararray import count
sys.path.append("/Users/zhanghangsheng/Documents/my_code/Cyber-Security-ML-Toolbox")
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from tests.image.get_interaction_single_i import get_average_pairwise_interaction_i
from tests.image.get_interaction import get_average_pairwise_interaction

import torch
import numpy as np
import random
from tqdm import trange
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import copy
from csmt.utils import load_mnist
from csmt.estimators.classification.pytorch import PyTorchClassifier

from tests.image.util import clamp
from tests.image.util import normalize_by_pnorm
from tests.image.interaction_loss import (InteractionLoss, get_features,
                               sample_for_interaction)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
# 设置随机数种子
setup_seed(20)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv_1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=5, stride=1)
        self.conv_2 = nn.Conv2d(in_channels=4, out_channels=10, kernel_size=5, stride=1)
        self.fc_1 = nn.Linear(in_features=4 * 4 * 10, out_features=100)
        self.fc_2 = nn.Linear(in_features=100, out_features=10)

    def forward(self, x):
        x = F.relu(self.conv_1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv_2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 10)
        x = F.relu(self.fc_1(x))
        x = self.fc_2(x)
        return x

class LR(nn.Module):
    def __init__(self):
        super(LR, self).__init__()
        self.linear = nn.Linear(in_features=28*28,out_features=10)
    def forward(self,x):
        x=x.view(-1,1*28*28)
        x = self.linear(x)
        return x

class Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()
        self.conv_1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=5, stride=1)
        self.fc_1 = nn.Linear(in_features=4 * 12 * 12, out_features=100)
        self.fc_2 = nn.Linear(in_features=100, out_features=10)

    def forward(self, x):
        x = F.relu(self.conv_1(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 12 * 12)
        x = F.sigmoid(self.fc_1(x))
        x = self.fc_2(x)
        return x
        
class MLP(nn.Module):
    def __init__(self):
        super(MLP,self).__init__()
        self.fc_1=nn.Linear(in_features=784,out_features=128)
        self.fc_2=nn.Linear(in_features=128,out_features=64)
        self.fc_3=nn.Linear(in_features=64,out_features=10)
    def forward(self,x):
        x = x.view(-1, 1 * 28 * 28)
        x = F.relu(self.fc_1(x))
        x = F.relu(self.fc_2(x))
        x = self.fc_3(x)
        return x

def train(X_train,y_train,model,batch_size,nb_epochs):
    loss_fun= nn.CrossEntropyLoss()
    # loss_fun=nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    model.train()
    num_batch = int(np.ceil(len(X_train) / float(batch_size)))
    ind = np.arange(len(X_train))
    for _ in range(nb_epochs):
        # random.shuffle(ind)
        # Train for one epoch
        with trange(num_batch) as t:
            for m in t:
                t.set_description('Train %i' %m)
                i_batch = torch.from_numpy(X_train[ind[m * batch_size : (m + 1) * batch_size]]).to(device)
                o_batch = torch.from_numpy(y_train[ind[m * batch_size : (m + 1) * batch_size]]).to(device)
                # Zero the parameter gradients
                optimizer.zero_grad()
                # Perform prediction
                model_outputs = model(i_batch)
                # print(model(i_batch))
                # print(o_batch)
                loss = loss_fun(model_outputs, o_batch)
                if m%100==0:
                    t.set_postfix(loss=loss.item())
                loss.backward()
                optimizer.step()
    return model

def print_result(X_test,y_test,model):
    model.eval()
    y_pred_torch= model(X_test)
    y_pred=y_pred_torch.detach().cpu().numpy()
    y_test=y_test.detach().cpu().numpy()
    y_pred=np.argmax(y_pred, axis=1) 
    print(y_pred)
    accuracy = np.sum(y_pred== y_test) / len(y_test)
    print("Accuracy on benign test examples: {}%".format(accuracy * 100))
    
def predict(X_test,y_test,model):
    model.eval()
    y_pred_torch= model(X_test)
    y_pred=y_pred_torch.detach().cpu().numpy()
    # accuracy = np.sum(np.argmax(y_pred, axis=1) == y_test) / len(y_test)
    # print("Accuracy on benign test examples: {}%".format(accuracy * 100))
    return np.argmax(y_pred, axis=1)[0]
    
def FGSM(X,y,model):
    epsilon=0.2
    loss_fn = nn.CrossEntropyLoss()
    delta = torch.zeros_like(X)
    delta.requires_grad_()
    loss = loss_fn(model(X + delta), y)
    loss.backward()
    cur_grad=delta.grad.data
    adv_X=X.data+epsilon*cur_grad.sign()
    return adv_X

def PGD(X, y,model,lam):
    num_steps=20
    step_size=0.01
    epsilon=1
    ord=np.inf
    loss_fn = nn.CrossEntropyLoss()
    delta = torch.zeros_like(X)
    delta.requires_grad_()
    grad = torch.zeros_like(X)
    deltas = torch.zeros_like(X).repeat(num_steps, 1, 1, 1)
    for i in range(num_steps):
        loss1 = loss_fn(model(X + delta), y)
        if lam != 0:  # Interaction-reduced attack
            average_pairwise_interaction=get_average_pairwise_interaction(model,X,y,delta)
            if i==99:
                print('zhs'+str(average_pairwise_interaction.item()))
            loss2 = -lam * average_pairwise_interaction
            loss = loss1 + loss2
        else:
            loss = loss1
        loss.backward()
        deltas[i, :, :, :] = delta.data
        cur_grad = delta.grad.data

        grad =cur_grad
        if ord == np.inf:
            delta.data += step_size* grad.sign()
            delta.data = clamp(delta.data, -epsilon, epsilon)
            delta.data = clamp(X.data + delta.data, 0.0, 1.0) - X.data
        delta.grad.data.zero_()
    adv_X = X.data + deltas
    return adv_X,deltas
        
    
(X_train, y_train), (X_test, y_test), min_pixel_value, max_pixel_value = load_mnist()
# print(X_test.shape)
X_train, y_train, X_test, y_test=X_train[0:10000], y_train[0:10000], X_test, y_test
X_train = np.transpose(X_train, (0, 3, 1, 2)).astype(np.float32)
X_test = np.transpose(X_test, (0, 3, 1, 2)).astype(np.float32)

y_train = np.argmax(y_train, axis=1)
y_test = np.argmax(y_test, axis=1)

X_test=torch.from_numpy(X_test).to(device)
y_test=torch.from_numpy(y_test).to(device)

model = Net()
model2 = MLP()
# model2=LR()
# model2 = Net2()
model=train(X_train,y_train,model,batch_size=64,nb_epochs=1)
model2=train(X_train,y_train,model2,batch_size=64,nb_epochs=1)
print_result(X_test,y_test,model)
print_result(X_test,y_test,model2)

# for i in range(10):
#     X=X_test[i:i+1]
#     y=y_test[i:i+1]
#     adv_X,deltas=PGD(X,y,model,lam=3)
#     delta=deltas[99:100]
#     # average_pairwise_interaction=get_average_pairwise_interaction(model,X,y,delta)
#     # print(average_pairwise_interaction.item())
#     pairwise_arr=[]
#     for con_i in range(14*14):
#         average_pairwise_interaction=get_average_pairwise_interaction_i(model,X,y,delta,con_i)
#         pairwise_arr.append(average_pairwise_interaction.item())
#     # print(np.array(pairwise_arr).mean()*14)
    
#     # np_pair=np.array(pairwise_arr).reshape(14,14)
#     # plt.matshow(np_pair)
#     # plt.clim(0, 1)
#     # plt.show()

    


count1=0
count2=0
for i in range(100):
    adv_X,deltas=PGD(X_test[i:i+1],y_test[i:i+1],model,lam=0)
    # plt.matshow(X_test[i:i+1].reshape((28, 28)))
    # plt.matshow(adv_X[98:99].reshape((28, 28)))
    # plt.clim(0, 1)
    # plt.show()
    print(count1)
    # dis=adv_X[98:99].reshape(28, 28).detach().cpu().numpy()-X_test[i:i+1].reshape(28, 28).detach().cpu().numpy()
    # print(np.linalg.norm(dis,ord=2))
    adv_y=predict(X_test[i:i+1],y_test[i:i+1],model)
    model1_y=predict(adv_X[19:20],y_test[i:i+1],model)
    model2_y=predict(adv_X[19:20],y_test[i:i+1],model2)
    if adv_y!=model1_y:
        count1+=1
    if adv_y!=model2_y:
        count2+=1
print(count1,count2)







