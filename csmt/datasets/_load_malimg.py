'''
Author: your name
Date: 2021-05-28 19:14:55
LastEditTime: 2021-05-28 19:22:37
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /Cyber-Security-ML-Toolbox/csmt/datasets/_load_drebin.py
'''
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import pandas as pd
import numpy as np
from os import path
import gzip
from csmt.datasets._base import get_mask,get_true_mask,add_str,get_dict
def load_malimg():
    # numpy_z=np.load('csmt/datasets/data/malimg/malimg_dataset_32x32.npy',allow_pickle=True)
    # print(numpy_z.shape)
    label_map={2:0,0:1}
    for i in range(25):
        if i==2:
            label_map[i]=0
        else:
            label_map[i]=1
    dataset=np.load('csmt/datasets/data/malimg/malimg.npz',allow_pickle=True)

    features = dataset["arr"][:, 0]
    features = np.array([feature for feature in features])
    features = np.reshape(
        features, (features.shape[0], features.shape[1] * features.shape[2])
    )
    labels = dataset["arr"][:, 1]
    labels = np.array([label for label in labels])
    X=features
    y=labels
    mask=get_true_mask([column for column in X])

    X=pd.DataFrame(X)
    y=pd.DataFrame(y)
    df= pd.concat([X,y],axis=1)
    df.iloc[:,-1]=df.iloc[:,-1].map(label_map)

    # print(df.iloc[:,-1].value_counts())
    X=df.iloc[:,0:-1]
    y=df.iloc[:,-1]
    mask=get_true_mask([column for column in X])
    return X,y,mask

