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
def load_twitter_spam():
    train_dataset_path= 'csmt/datasets/data/twitter_spam/twitter_spam_reduced.train.csv'
    test_dataset_path = 'csmt/datasets/data/twitter_spam/twitter_spam_reduced.test.csv'
    model_file='csmt/datasets/pickles/twitter_spam_dataframe.pkl'
    if path.exists(model_file):
        df = pd.read_pickle(model_file)
        X=df.iloc[:,1:]
        mask=get_true_mask([column for column in X])
        y=df.iloc[:,0]
        return X,y,mask
    df_train = pd.read_csv(train_dataset_path, header=0)
    df_test = pd.read_csv(test_dataset_path, header=0)

    df_train = df_train.sample(frac=0.02, random_state=20)
    # df_train.to_pickle(model_file)

    X=df_train.iloc[:,1:]
    y=df_train.iloc[:,0]
    # print(df_test.iloc[:,0].value_counts())
    # print(y)
    
    mask=get_true_mask([column for column in X])
    return X,y,mask

