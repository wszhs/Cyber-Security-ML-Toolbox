'''
Author: your name
Date: 2021-04-19 11:27:47
LastEditTime: 2021-07-10 19:35:49
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /Cyber-Security-ML-Toolbox/csmt/classifiers/classic/hidden_markov_model.py
'''
from hmmlearn import hmm
import numpy as np
from csmt.classifiers.abstract_model import AbstractModel
from csmt.classifiers.classic.hmm_classifier import HMM_classifier
from sklearn.preprocessing import MinMaxScaler
import pickle

class HMM(AbstractModel):
    def __init__(self,input_size,output_size):
        self.model=HMM_classifier(hmm.GaussianHMM())
    
    def train(self, X_train, y_train):
        X_train=np.reshape(X_train,(X_train.shape[0],X_train.shape[1],1))
        self.model.fit(X_train, y_train)

    # def predict(self, X):
    #     X=np.reshape(X,(X.shape[0],X.shape[1],1))
    #     pred_np=np.zeros(X.shape[0])
    #     for i in range(0,X.shape[0]):
    #         pred = self.model.predict(X[i])
    #         pred_np[i]=pred
    #     return pred_np
    
    def predict(self,X):
        #实现有错误
        X=np.reshape(X,(X.shape[0],X.shape[1],1))
        pred_pro_np=np.zeros((X.shape[0],2))
        for i in range(0,X.shape[0]):
            pred_pro=np.zeros(2)
            pred_pro[0] = self.model.predict_proba(X[i])[0]
            pred_pro[1]=self.model.predict_proba(X[i])[1]
            pred_pro_np[i]=pred_pro
        scaler = MinMaxScaler().fit(pred_pro_np)
        pred_pro_np = scaler.transform(pred_pro_np)
        return pred_pro_np

    def save(self, path):
        pickle.dump(self.model, open(path,'wb'))

    def load(self, path):
        self.model=pickle.load(open(path,'rb'))