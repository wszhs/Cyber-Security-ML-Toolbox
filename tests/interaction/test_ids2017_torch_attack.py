'''
Author: your name
Date: 2021-06-24 15:56:14
LastEditTime: 2021-07-23 11:50:50
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
from numpy.lib.twodim_base import mask_indices
from sklearn import preprocessing
from sklearn import model_selection
sys.path.append("/Users/zhanghangsheng/Documents/my_code/Cyber-Security-ML-Toolbox")
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

import torch
import numpy as np
import random
from tqdm import trange
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import copy

from csmt.utils import load_mnist
from csmt.datasets import load_breast_cancer_zhs
from csmt.datasets import load_cicids2017
from csmt.datasets import load_contagiopdf
from csmt.datasets import load_cicandmal2017
from csmt.datasets import load_kitsune
from csmt.estimators.classification.pytorch import PyTorchClassifier

from tests.interaction.util import clamp
from tests.interaction.util import normalize_by_pnorm
from tests.interaction.get_interaction import get_average_pairwise_interaction
from tests.interaction.get_interaction_single_i import get_average_pairwise_interaction_i
from csmt.estimators.classification.scikitlearn import SklearnClassifier
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
# 设置随机数种子
setup_seed(20)

class Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()
        self.conv_1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=5, stride=1)
        self.fc_1 = nn.Linear(in_features= 4* 3 * 6, out_features=10)
        self.fc_2 = nn.Linear(in_features=10, out_features=2)

    def forward(self, x):
        x=x.view(-1,1,7,10)
        x = F.relu(self.conv_1(x))
        x = x.view(-1, 4 * 3 * 6)
        x = F.relu(self.fc_1(x))
        x = self.fc_2(x)
        return x

class MLP(nn.Module):
    def __init__(self):
        super(MLP,self).__init__()
        self.fc_1=nn.Linear(in_features=70,out_features=32)
        self.fc_2=nn.Linear(in_features=32,out_features=10)
        self.fc_3=nn.Linear(in_features=10,out_features=2)
    def forward(self,x):
        x = F.relu(self.fc_1(x))
        x = F.relu(self.fc_2(x))
        x = self.fc_3(x)
        return x


def train(X_train,y_train,model,batch_size,nb_epochs):
    loss_fun= nn.CrossEntropyLoss()
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

def classic_train(X_train,y_train):
    # model=SVC()
    model=DecisionTreeClassifier()
    # model= linear_model.LogisticRegression(max_iter=1000)
    # model=RandomForestClassifier()
    # model=KNeighborsClassifier(n_neighbors=6)
    art_classifier = SklearnClassifier(model=model)
    art_classifier.fit(X_train,y_train)
    return art_classifier

def print_result(X_test,y_test,model):
    model.eval()
    y_pred_torch= model(X_test)
    y_pred=y_pred_torch.detach().cpu().numpy()
    y_test=y_test.detach().cpu().numpy()
    y_pred=np.argmax(y_pred, axis=1) 
    print(y_pred)
    accuracy = np.sum(y_pred== y_test) / len(y_test)
    print("Accuracy on benign test examples: {}%".format(accuracy * 100))

def print_result_classic(X_test,y_test,model):
    y_test=y_test.detach().cpu().numpy()
    y_pred=np.argmax(model.predict(X_test), axis=1)
    accuracy = np.sum(y_pred== y_test) / len(y_test)
    print("Accuracy on benign test examples: {}%".format(accuracy * 100))

def predict_classic(X_test,y_test,model):
    y_test=y_test.detach().cpu().numpy()
    y_pred=model.predict(X_test)
    return np.argmax(y_pred, axis=1)[0]


def predict(X_test,y_test,model):
    model.eval()
    y_pred_torch= model(X_test)
    y_pred=y_pred_torch.detach().cpu().numpy()
    return np.argmax(y_pred, axis=1)[0]

def PGD(X, y,model,lam):
    num_steps=10
    step_size=0.1
    epsilon=0.3
    grid_scale=70
    sample_grid_num=50
    times=30
    ord=np.inf
    loss_fn = nn.CrossEntropyLoss()
    delta = torch.zeros_like(X)
    delta.requires_grad_()
    grad = torch.zeros_like(X)
    deltas = torch.zeros_like(X).repeat(num_steps, 1)
    for i in range(num_steps):
        loss1 = loss_fn(model(X + delta), y)
        if lam != 0:  # Interaction-reduced attack
            average_pairwise_interaction=get_average_pairwise_interaction(model,X,y,delta,sample_grid_num=sample_grid_num,grid_scale=grid_scale,times=times)
            loss2 = -lam * average_pairwise_interaction
            loss = loss1 + loss2
        else:
            loss = loss1
        loss.backward()
        deltas[i, :] = delta.data
        cur_grad = delta.grad.data

        grad =cur_grad
        if ord == np.inf:
            delta.data += step_size* grad.sign()
            delta.data = clamp(delta.data, -epsilon, epsilon)
            delta.data = clamp(X.data + delta.data, 0.0, 1.0) - X.data
        delta.grad.data.zero_()
    adv_X = X.data + deltas
    return adv_X,deltas
        
X,y,mask=load_cicids2017()
scaler= preprocessing.MinMaxScaler().fit(X)
X = scaler.transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
y_train, y_test=y_train.values, y_test.values

X_train=X_train.astype(np.float32)
X_test=X_test.astype(np.float32)
y_train=y_train.astype(np.int64)
y_test=y_test.astype(np.int64)

X_test=torch.from_numpy(X_test).to(device)
y_test=torch.from_numpy(y_test).to(device)

model = Net2()
model=train(X_train,y_train,model,batch_size=64,nb_epochs=1)
model2=classic_train(X_train,y_train)
print_result(X_test,y_test,model)
print_result_classic(X_test,y_test,model2)




count1=0
count2=0
for i in range(100):
    adv_X,deltas=PGD(X_test[i:i+1],y_test[i:i+1],model,lam=0)
    print(count1)
    adv_y=predict(X_test[i:i+1],y_test[i:i+1],model)
    model1_y=predict(adv_X[9:10],y_test[i:i+1],model)
    model2_y=predict_classic(adv_X[9:10],y_test[i:i+1],model2)
    if adv_y!=model1_y:
        count1+=1
    if adv_y!=model2_y:
        count2+=1
print(count1,count2)

# count1=0
# count2=0
# for i in range(100):
#     adv_X,deltas=PGD(X_test[i:i+1],y_test[i:i+1],model,lam=5)
#     print(count1)
#     adv_y=predict(X_test[i:i+1],y_test[i:i+1],model)
#     model1_y=predict(adv_X[9:10],y_test[i:i+1],model)
#     model2_y=predict_classic(adv_X[9:10],y_test[i:i+1],model2)
#     if adv_y!=model1_y:
#         count1+=1
#     if adv_y!=model2_y:
#         count2+=1
# print(count1,count2)

# count1=0
# count2=0
# for i in range(100):
#     adv_X,deltas=PGD(X_test[i:i+1],y_test[i:i+1],model,lam=0.3)
#     print(count1)
#     adv_y=predict(X_test[i:i+1],y_test[i:i+1],model)
#     model1_y=predict(adv_X[9:10],y_test[i:i+1],model)
#     model2_y=predict_classic(adv_X[9:10],y_test[i:i+1],model2)
#     if adv_y!=model1_y:
#         count1+=1
#     if adv_y!=model2_y:
#         count2+=1
# print(count1,count2)








