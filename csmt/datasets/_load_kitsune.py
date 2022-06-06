
'''
Author: your name
Date: 2021-03-25 14:30:40
LastEditTime: 2021-07-05 14:47:40
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /NashHE/nashhe/datasets/_loadcic.py
'''
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from csmt.datasets._base import get_mask,get_true_mask,add_str,get_dict
import pickle
import pandas as pd
import numpy as np
from os import path
from csmt.decomposition.decomposition import tsne_dim_redu

def load_kitsune():

    # X = np.load('csmt/datasets/data/Kitsune/mirai/kitsune_feature_data.npy')
    X=np.load('csmt/datasets/data/Kitsune/mirai/kitsune.npy')
    # print(X.shape)
    y= np.load('csmt/datasets/data/Kitsune/mirai/labels.npy')

    # import_index=np.array([62,59,0,15,65,72,22,3,79,29]) 
    # X=X[:,import_index]
    # X=X[:,0:10]

    X=pd.DataFrame(X)
    y=pd.DataFrame(y)
    df= pd.concat([X,y],axis=1)
    df_1=df[df.iloc[:,-1]==1]
    df_0=df[df.iloc[:,-1]==0]

    # df_0 = df_0[0:50000]
    # df_1 = df_1[0:50000]

    # df_0 = df_0.sample(frac=0.01, random_state=20)
    # df_1 = df_1.sample(frac=0.01, random_state=20)

    print('0-'+str(df_0.shape[0]))
    print('1-'+str(df_1.shape[0]))

    df= pd.concat([df_0,df_1])
    X=df.iloc[:,0:-1]
    y=df.iloc[:,-1]

    mask=get_true_mask([column for column in X])
    return X,y,mask

def load_kitsune_1():

    X = np.load('csmt/datasets/data/Kitsune/SYN_DoS/kitsune_feature_data.npy')
    # X = np.load('csmt/datasets/data/Kitsune/SYN_DoS/syn_dos.npy')
    y= np.load('csmt/datasets/data/Kitsune/SYN_DoS/labels.npy')
    X=pd.DataFrame(X)
    y=pd.DataFrame(y)
    df= pd.concat([X,y],axis=1)
    # df.to_csv('csmt/datasets/data/Kitsune/mirai/mirai.csv')
    df_1=df[df.iloc[:,-1]==1]
    df_0=df[df.iloc[:,-1]==0]
    df_0=df_0.iloc[0:20000]

    # df_0 = df_0.sample(frac=0.005, random_state=20)
    # df_1 = df_1.sample(frac=0.5, random_state=20)

    print('0-'+str(df_0.shape[0]))
    print('1-'+str(df_1.shape[0]))

    df= pd.concat([df_0,df_1])
    X=df.iloc[:,0:-1]
    y=df.iloc[:,-1]

    mask=get_true_mask([column for column in X])
    return X,y,mask

def load_kitsune_1():

    X = np.load('csmt/datasets/data/Kitsune/Fuzzing/kitsune_feature_data.npy')
    y= np.load('csmt/datasets/data/Kitsune/Fuzzing/labels.npy')
    X=pd.DataFrame(X)
    y=pd.DataFrame(y)
    df= pd.concat([X,y],axis=1)
    # df.to_csv('csmt/datasets/data/Kitsune/mirai/mirai.csv')
    df_1=df[df.iloc[:,-1]==1]
    df_0=df[df.iloc[:,-1]==0]
    df_0 = df_0.sample(frac=0.003, random_state=20)
    df_1 = df_1.sample(frac=0.005, random_state=20)

    print('0-'+str(df_0.shape[0]))
    print('1-'+str(df_1.shape[0]))

    df= pd.concat([df_0,df_1])
    X=df.iloc[:,0:-1]
    y=df.iloc[:,-1]

    mask=get_true_mask([column for column in X])
    return X,y,mask

def load_kitsune_1():

    X = np.load('csmt/datasets/data/Kitsune/SSDP_Flood/kitsune_feature_data.npy')
    y= np.load('csmt/datasets/data/Kitsune/SSDP_Flood/labels.npy')
    X=pd.DataFrame(X)
    y=pd.DataFrame(y)
    df= pd.concat([X,y],axis=1)
    # df.to_csv('csmt/datasets/data/Kitsune/mirai/mirai.csv')
    df_1=df[df.iloc[:,-1]==1]
    df_0=df[df.iloc[:,-1]==0]
    df_0 = df_0.sample(frac=0.003, random_state=20)
    df_1 = df_1.sample(frac=0.005, random_state=20)

    print('0-'+str(df_0.shape[0]))
    print('1-'+str(df_1.shape[0]))

    df= pd.concat([df_0,df_1])
    X=df.iloc[:,0:-1]
    y=df.iloc[:,-1]

    mask=get_true_mask([column for column in X])
    return X,y,mask

def load_kitsune_old():
    test_data = np.load('/Users/zhanghangsheng/Documents/my_code/Cyber-Security-ML-Toolbox/csmt/datasets/data/Kitsune/test.npy')
    train_ben = np.load('/Users/zhanghangsheng/Documents/my_code/Cyber-Security-ML-Toolbox/csmt/datasets/data/Kitsune/train_ben.npy')
    # test_data_pd=pd.DataFrame(test_data)
    # train_ben_pd=pd.DataFrame(train_ben)

    # test_data_pd['label'] =1
    # train_ben_pd['label'] =0
    # df= pd.concat([test_data_pd,train_ben_pd])
    # df = df.sample(frac=0.8, random_state=20)
    # # print(df.shape)
    
    # X = df.drop(['label'], axis=1)
    # y = df['label']
    return train_ben,test_data

def load_kitsune_old():
    X_file_path='csmt/datasets/data/Kitsune/mirai/Mirai_dataset.csv'
    y_file_path='csmt/datasets/data/Kitsune/mirai/mirai_labels.csv'
    model_file='csmt/datasets/pickles/mirai_dataframe.pkl'

    if path.exists(model_file):
        df = pd.read_pickle(model_file)
        X=df.iloc[:, 1:-2]
        y=df.iloc[:,-1]
        mask=get_true_mask([column for column in X])
        return X,y,mask

    X = pd.read_csv(X_file_path, encoding='utf8', low_memory=False)
    y = pd.read_csv(y_file_path, encoding='utf8', low_memory=False)
    df=pd.concat([X,y],axis=1)
    df = df.sample(frac=0.005, random_state=20)
    print(df.shape)
    X=df.iloc[:, 1:-2]
    y=df.iloc[:,-1]
    df.to_pickle(model_file)
    mask=get_true_mask([column for column in X])
    return X,y,mask