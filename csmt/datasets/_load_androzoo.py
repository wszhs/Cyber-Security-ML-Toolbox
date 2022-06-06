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

def load_androzoo():
    X_path='csmt/datasets/data/androzoo/androzoo/derbin/X_1.pkl'
    y_path='csmt/datasets/data/androzoo/androzoo/derbin/y_1.pkl'
    X = pd.read_pickle(X_path)
    X=X.iloc[:,:100]
    y=pd.read_pickle(y_path)
    mask=get_true_mask([column for column in X])
    print(X)
    return X,y,mask

