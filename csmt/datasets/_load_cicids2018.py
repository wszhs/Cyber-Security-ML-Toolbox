'''
Author: your name
Date: 2021-03-25 14:30:40
LastEditTime: 2021-07-18 13:32:36
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /NashHE/nashhe/datasets/_loadcic.py
'''

import pandas as pd
import numpy as np
from os import path
from csmt.datasets._base import get_mask,get_true_mask,add_str,get_dict

def load_cicids2018():
    data = np.load('csmt/datasets/data/CIC-IDS-2018/IDS_new_Infilteration.npz')
    X_train=data['X_train']
    X_test=data['X_test']
    y_train=data['y_train']
    y_test=data['y_test']

    X=np.concatenate((X_train,X_test),axis=0)
    y=np.concatenate((y_train,y_test),axis=0)
    y[y!=0]=1

    X=pd.DataFrame(X)
    y=pd.DataFrame(y)
    df= pd.concat([X,y],axis=1)
    df = df.sample(frac=0.9, random_state=20)
    X=df.iloc[:,0:-1]
    y=df.iloc[:,-1]
    mask=get_true_mask([column for column in X])
    return X,y,mask