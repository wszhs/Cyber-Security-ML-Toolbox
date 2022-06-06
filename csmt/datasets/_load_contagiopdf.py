'''
Author: your name
Date: 2021-03-25 14:31:34
LastEditTime: 2021-07-12 16:14:18
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /NashHE/nashhe/datasets/_load_.py
'''

from csmt.utils import make_directory
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import pickle
import pandas as pd
import numpy as np
from os import path
from csmt.datasets._base import get_mask,get_true_mask,add_str,get_dict

def load_contagiopdf():
    file_path='csmt/datasets/data/ContagioPDF/ContagioPDFData.csv'
    df = pd.read_csv(file_path, encoding='utf8', low_memory=False)
    df = df.drop(['filename'], axis=1)
    df.columns = df.columns.str.lstrip()

    df = df.dropna()
    df = df.reset_index(drop=True)

    df['class'] = df['class'].astype(str).map({'False': 0, 'True': 1})
    df['box_other_only'] = df['box_other_only'].astype(str).map({'False': 0, 'True': 1})
    df['pdfid_mismatch'] = df['pdfid_mismatch'].astype(str).map({'False': 0, 'True': 1})

    df = df.drop_duplicates()
    df = df.sample(frac=0.80, random_state=20)
    X = df.drop(['class'], axis=1)
    X=df.iloc[:,0:2]
    y = df['class']
    mask=get_true_mask([column for column in X])
    return X,y,mask

def load_mimi_contagiopdf():
        
    file_path='csmt/datasets/data/ContagioPDF/ContagioPDFData.csv'
    df = pd.read_csv(file_path, encoding='utf8', low_memory=False)
    df = df.drop(['filename'], axis=1)
    df.columns = df.columns.str.lstrip()

    df = df.dropna()
    df = df.reset_index(drop=True)

    df['class'] = df['class'].astype(str).map({'False': 0, 'True': 1})
    df['box_other_only'] = df['box_other_only'].astype(str).map({'False': 1, 'True': 0})
    df['pdfid_mismatch'] = df['pdfid_mismatch'].astype(str).map({'False': 1, 'True': 0})

    df = df.drop_duplicates()
    df_train, df_test= train_test_split(df,test_size=0.2, random_state=42)


    train_normal_df=df_train[df_train['class']==1]
    train_malicious_df=df_train[df_train['class']==0]

    train_normal_traffic = train_normal_df.drop(['class'], axis=1)
    train_normal_labels = train_normal_df['class']

    train_malicious_traffic = train_malicious_df.drop(['class'], axis=1)
    train_malicious_labels = train_malicious_df['class']

    trainingset=(train_normal_traffic,train_malicious_traffic,train_normal_labels,train_malicious_labels)

    test_normal_df=df_test[df_test['class']==1]
    test_malicious_df=df_test[df_test['class']==0]

    test_normal_traffic = test_normal_df.drop(['class'], axis=1)
    test_normal_labels = test_normal_df['class']

    test_malicious_traffic = test_malicious_df.drop(['class'], axis=1)
    test_malicious_labels = test_malicious_df['class']

    testingset=(test_normal_traffic,test_malicious_traffic,test_normal_labels,test_malicious_labels)

    return train_normal_traffic.shape[1],trainingset,testingset
