
from csmt.datasets import *

from csmt.normalizer import Normalizer
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from csmt.decomposition.decomposition import tsne_dim_redu,pca_dim_redu,svd_dim_redu
from sklearn.model_selection import train_test_split
from csmt.classifiers.graph.dataset import Dataset,CogDLDataset

import numpy as np


def get_graph_cogdl_datasets(options):
    datasets_name=options.datasets
    data=globals().get('load_'+datasets_name)()
    return data

def get_graph_datasets(options):
    datasets_name=options.datasets
    features, adj, labels, split_ids=globals().get('load_'+datasets_name)()
    return features, adj, labels, split_ids

def get_graph_grb_datasets(options):
    dataset_name=options.datasets
    if 'grb' in dataset_name:
        data = Dataset(name=dataset_name, 
                    data_dir="./data/",
                    mode="full",
                    feat_norm="arctan")
    else:
        data=CogDLDataset(name=dataset_name)
    return data

def get_raw_datasets(options):
    datasets_name=options.datasets
    X,y,mask=globals().get('load_'+datasets_name)()
    # if type(y) is not np.ndarray:
    #     X,y=X.values,y.values
    return X,y,mask

def get_datasets(options):
    datasets_name=options.datasets
    if datasets_name=='mnist':
        X_train,y_train,X_val,y_val,X_test,y_test,mask=load_mnist()
        return X_train,y_train,X_val,y_val,X_test,y_test,mask
    if datasets_name=='wfa':
        X_train,y_train,X_val,y_val,X_test,y_test,mask=load_wfa()
        return X_train,y_train,X_val,y_val,X_test,y_test,mask
    if datasets_name=='pcap':
        # from csmt.datasets import load_pcap
        X_train,y_train,X_val,y_val,X_test,y_test,mask=load_pcap()
        return X_train,y_train,X_val,y_val,X_test,y_test,mask
    #加载数据集
    # from csmt.datasets import load_nslkdd
    X,y,mask=globals().get('load_'+datasets_name)()
    normer = Normalizer(X.shape[-1],online_minmax=False)
    X = normer.fit_transform(X)

    return pre_processing(X,y,mask)

def pre_processing(X,y,mask):
    # mm=MinMaxScaler()
    # X=mm.fit_transform(X)
    # X_train,y_train,X_val,y_val,X_test,y_test=train_val_test_split(X,y,0.6,0.3,0.1)
    # X_train, X_test, y_train, y_test = X,X,y,y
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4,random_state=42)
    X_val,y_val=X_test,y_test

    X_train=X_train.astype(np.float32)
    X_test=X_test.astype(np.float32)
    X_val=X_val.astype(np.float32)
    y_train=y_train.astype(np.int32)
    y_test=y_test.astype(np.int32)
    y_val=y_val.astype(np.int32)

    if type(y_train) is not np.ndarray:
        X_train,y_train,X_val,y_val,X_test,y_test=X_train.values,y_train.values,X_val.values,y_val.values,X_test.values,y_test.values

    return X_train,y_train,X_val,y_val,X_test,y_test,mask

# def train_val_test_split(X,y, ratio_train, ratio_test, ratio_val):
#     X_train,X_middle,y_train,y_middle = train_test_split(X,y,train_size=ratio_train, test_size=ratio_test + ratio_val,random_state=42,shuffle=True)
#     ratio = ratio_val/(1-ratio_train)
#     X_test,X_val,y_test,y_val = train_test_split(X_middle,y_middle,test_size=ratio,random_state=42,shuffle=True)
#     return X_train,y_train,X_val,y_val,X_test,y_test