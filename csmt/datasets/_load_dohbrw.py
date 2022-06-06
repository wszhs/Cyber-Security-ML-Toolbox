'''
Author: your name
Date: 2021-04-24 10:14:19
LastEditTime: 2021-05-19 11:17:24
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /Cyber-Security-ML-Toolbox/csmt/datasets/_load_dohbrw.py
'''
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import pickle
import pandas as pd
import numpy as np
from os import path
from csmt.datasets._base import get_mask,get_true_mask,add_str,get_dict

def load_dohbrw(data_class_type):

    if data_class_type=='single_binary' or data_class_type=='multi':
        raise Exception('Data classification type is not supported！')
    file_path='csmt/datasets/data/CIC-DoHBrw-2020/CicFlowMeter-Doh.csv'
    model_file='csmt/datasets/pickles/dohbrw_dataframe.pkl'

    if path.exists(model_file):
        df = pd.read_pickle(model_file)
        X = df.drop(['label'], axis=1)
        mask=get_true_mask([column for column in X])
        y = df['label']
        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
        return X_train,y_train,X_test,y_test,mask

    feature_array=['Duration','FlowBytesSent','FlowSentRate','FlowBytesReceived','FlowReceivedRate','PacketLengthVariance',
                    'PacketLengthStandardDeviation','PacketLengthMean','PacketLengthMedian','PacketLengthMode','PacketLengthSkewFromMedian',
                    'PacketLengthSkewFromMode','PacketLengthCoefficientofVariation','PacketTimeVariance','PacketTimeStandardDeviation',
                    'PacketTimeMean','PacketTimeMedian','PacketTimeMode','PacketTimeSkewFromMedian','PacketTimeSkewFromMode','PacketTimeCoefficientofVariation',
                    'ResponseTimeTimeVariance','ResponseTimeTimeStandardDeviation','ResponseTimeTimeMean','ResponseTimeTimeMedian','ResponseTimeTimeMode',
                    'ResponseTimeTimeSkewFromMedian','ResponseTimeTimeSkewFromMode','ResponseTimeTimeCoefficientofVariation','label']

    label_map={'Chrome':0,'Firefox':0,'dns2tcp':1,'dnscat2':1,'iodine':1}

    df = pd.read_csv(file_path, encoding='utf8', low_memory=False)

    df=df[feature_array]

    # remove white space at the beginning of string in dataframe header
    df.columns = df.columns.str.lstrip()

    df['label'] = df['label'].map(label_map)
    df.drop(df[np.isnan(df['ResponseTimeTimeMedian'])].index, inplace=True)
    df.drop(df[np.isnan(df['ResponseTimeTimeSkewFromMedian'])].index, inplace=True)

    df = df.sample(frac=0.01, random_state=20)
    print(df['label'].value_counts())

    X = df.drop(['label'], axis=1)
    mask=get_true_mask([column for column in X])
    y = df['label']
    print(y.value_counts())
    return X,y,mask