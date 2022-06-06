import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
from csmt.classifiers.abstract_model import AbstractModel
batch_size = 128 
lr = 1e-3 
weight_decay = 1e-6
epoches = 5 

def se2rmse(a):
    return torch.sqrt(sum(a.t())/a.shape[1])

class autoencoder(nn.Module):
    def __init__(self, feature_size):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(feature_size, int(feature_size*0.75)),
                                     nn.ReLU(True),
                                     nn.Linear(int(feature_size*0.75), int(feature_size*0.5)),
                                     nn.ReLU(True),
                                     nn.Linear(int(feature_size*0.5),int(feature_size*0.25)),
                                     nn.ReLU(True),
                                     nn.Linear(int(feature_size*0.25),int(feature_size*0.1)))

        self.decoder = nn.Sequential(nn.Linear(int(feature_size*0.1),int(feature_size*0.25)),
                                     nn.ReLU(True),
                                     nn.Linear(int(feature_size*0.25),int(feature_size*0.5)),
                                     nn.ReLU(True),
                                     nn.Linear(int(feature_size*0.5),int(feature_size*0.75)),
                                     nn.ReLU(True),
                                     nn.Linear(int(feature_size*0.75),int(feature_size)),
                                     )
    
    def forward(self, x):
        encode = self.encoder(x)
        decode = self.decoder(encode)
        return decode
    
criterion = nn.MSELoss()
getMSEvec = nn.MSELoss(reduction='none')

class AbAutoEncoder(AbstractModel):
    def __init__(self,input_size,output_size):
        self.feature_size=input_size
        self.AD_threshold=0
        self.model=None

    def train(self, X_train, y_train,X_val,y_val):
        X_train=X_train[y_train==0]
        y_train=y_train[y_train==0]
        self.model = autoencoder(self.feature_size)
        optimizier = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.model.train()

        X_train = torch.from_numpy(X_train)   
        if torch.cuda.is_available(): X_train = X_train.cuda()
        torch_dataset = Data.TensorDataset(X_train, X_train)
        loader = Data.DataLoader(
            dataset=torch_dataset,
            batch_size=batch_size,
            shuffle=True,
        )
        for epoch in range(epoches):
            for step, (batch_x, batch_y) in enumerate(loader):
                output = self.model(batch_x)
                loss = criterion(output, batch_y)
                optimizier.zero_grad()
                loss.backward()
                optimizier.step()
                if step % 100 == 0 :
                    print('epoch:{}/{}'.format(epoch,step), '|Loss:', loss.item())
        
        self.model.eval()
        output = self.model(X_train)
        mse_vec = getMSEvec(output,X_train)
        rmse_vec = se2rmse(mse_vec).cpu().data.numpy()

        # print("max AD score",max(rmse_vec))
        thres = max(rmse_vec)
        self.AD_threshold = thres
        rmse_vec.sort()
        pctg = 0.9999   # 99% percentage for threshold selection
        thres = rmse_vec[int(len(rmse_vec)*pctg)]
        print("thres:",thres)
    
    def predict(self, X_test):
        self.model.eval()
        X_test = torch.from_numpy(X_test)    
        X_test = X_test
        output = self.model(X_test)
        mse_vec = getMSEvec(output,X_test)
        rmse_vec = se2rmse(mse_vec).cpu().data.numpy()
        anomaly_scores=rmse_vec.reshape(-1,1)
        y_pred=np.hstack((-(anomaly_scores-self.AD_threshold),anomaly_scores-self.AD_threshold))
        return y_pred

    def save(self, path):
        return 0

    def load(self, path):
        return 0
