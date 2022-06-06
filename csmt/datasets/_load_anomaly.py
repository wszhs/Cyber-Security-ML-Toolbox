from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import pickle
import pandas as pd
import numpy as np
from os import path
from csmt.datasets._base import get_mask,get_true_mask,add_str,get_dict
from scipy.io import loadmat

def load_anomaly():
    mat = loadmat('csmt/datasets/data/Anomaly/ionosphere.mat')
    X = mat['X']
    y = mat['y'].ravel()
    # print(X)
    # print(X.shape)
    # print(y)
    mask=get_true_mask([column for column in X])
    return X,y,mask