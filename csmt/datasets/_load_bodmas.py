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

def load_bodmas():
    model_file='csmt/datasets/pickles/bodmas_dataframe.pkl'
    if path.exists(model_file):
        df = pd.read_pickle(model_file)
        X=df.iloc[:,0:-1]
        mask=get_true_mask([column for column in X])
        y=df.iloc[:,-1]
        return X,y,mask
    numpy_z=np.load('csmt/datasets/data/bodmas/bodmas.npz')
    X=numpy_z['X']
    y=numpy_z['y']
    X=pd.DataFrame(X)
    y=pd.DataFrame(y)
    df= pd.concat([X,y],axis=1)
    df = df.sample(frac=0.03, random_state=20)
    df.to_pickle(model_file)
    df_1=df[df.iloc[:,-1]==1]
    df_0=df[df.iloc[:,-1]==0]

    print('0-'+str(df_0.shape[0]))
    print('1-'+str(df_1.shape[0]))

    df= pd.concat([df_0,df_1])
    X=df.iloc[:,0:-1]
    y=df.iloc[:,-1]
    
    mask=get_true_mask([column for column in X])
    return X,y,mask

