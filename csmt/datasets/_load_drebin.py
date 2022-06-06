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
import pickle
import pandas as pd
import numpy as np
from os import path
import pickle
import gzip
from csmt.datasets._base import get_mask,get_true_mask,add_str,get_dict

def load_drebin():
    file_path='csmt/datasets/data/DrebinRed/drebin-reduced.tar.gz'
    with gzip.open(file_path, 'rb') as f_ref:
        # Loading and returning the object
        ds=pickle.load(f_ref, encoding='bytes')

    print("Num. samples: ", ds.num_samples)

    n_neg = sum(ds.Y == 0)
    n_pos = sum(ds.Y == 1)

    print("Num. benign samples: ", n_neg)
    print("Num. malicious samples: ", n_pos)

    print("Num. features: ", ds.num_features)

    # return X_train,y_train,X_test,y_test
    return ds.X,ds.Y