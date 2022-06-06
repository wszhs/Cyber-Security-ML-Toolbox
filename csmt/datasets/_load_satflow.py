'''
Author: your name
Date: 2021-03-25 14:27:29
LastEditTime: 2021-05-08 18:45:19
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /NashHE/nashhe/datasets/_load_cicandmal2017.py
'''
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import pickle
import pandas as pd
import numpy as np
from os import path
from csmt.datasets._base import get_mask,get_true_mask,add_str,get_dict

def load_satflow_old(data_class_type):
    if data_class_type=='multi':
        raise Exception('Data classification type is not supported！')
    elif data_class_type=='single_binary':
        return load_satflow_single()
    elif data_class_type=='all_binary':
        return load_satflow_all()

def load_satflow_single():
    file_path='csmt/datasets/data/Sat-Flow/Sat_Flow.csv'

    label_map = {'BENIGN': 0, 'Syn_DDoS': 1, 'UDP_DDoS': 1, 'Botnet': 1,'Web Attack': 1, 'Backdoor': 1,'LDAP_DDoS': 1, 'MSSQL_DDoS': 1, 'NetBIOS_DDoS': 1, 'Portmap_DDoS': 1}
    
    ##读入csv文件         
    df = pd.read_csv(file_path, encoding='utf8', low_memory=False)
    
    ##删除第一列序号，无用信息
    df = df.drop([df.columns[0]], axis=1)
    
    # 删除字符串左侧空格
    df.columns = df.columns.str.lstrip()
    
    ##删除空值
    df = df.dropna()
    df = df.reset_index(drop=True)
    
    ##按照Label,拆分成10个表,存储在split_table中
    attack_list = list(df['Label'].drop_duplicates())
    split_table = []
    for i in attack_list:
        split = df[df['Label'] == i ] 
        split_table.append(split)
    
    ##将split_table由list转换为series,并设置索引
    split_table = pd.Series(split_table) 
    split_table.index = ['Syn_DDoS','UDP_DDoS','Botnet','Web Attack','Backdoor','LDAP_DDoS','MSSQL_DDoS','NetBIOS_DDoS','Portmap_DDoS','BENIGN']
    
    ##删除BENIGN表
    del split_table['BENIGN'] 
   
    ##采样BENIGN
    benign = df[df['Label'] == 'BENIGN']
    benign1 = benign.sample(n = 55000,random_state = 1,axis = 0)  
    benign2 = benign.sample(n = 12000,random_state = 1,axis = 0) 
    benign3 = benign.sample(n = 15000,random_state = 1,axis = 0)  
    
    ##将split_table[]分别与BENIGN拼接
    split_table[0] = pd.concat([split_table[0],benign1])
    split_table[1] = pd.concat([split_table[1],benign1])
    split_table[2] = pd.concat([split_table[2],benign3])
    split_table[3] = pd.concat([split_table[3],benign2])
    split_table[4] = pd.concat([split_table[4],benign2])
    split_table[5] = pd.concat([split_table[5],benign3])
    split_table[6] = pd.concat([split_table[6],benign3])
    split_table[7] = pd.concat([split_table[7],benign2])
    split_table[8] = pd.concat([split_table[8],benign3])
    
    ##批量更改Label
    for i in split_table:
        i['Label'] = i['Label'].map(label_map)
    
    ##批量设置X，y    
    X = []
    y = []
    for i in split_table:
        a = i.drop(['Label'], axis=1)
        b = i['Label']
        X.append(a)
        y.append(b)
    mask=get_true_mask([column for column in X])
    ##批量生成测试集、训练集
    X_train = []
    X_test = []
    y_train = []
    y_test = []
    for (a,b) in zip(X,y):
    	a_train, a_test, b_train, b_test = train_test_split(a, b, stratify=b, test_size=0.2, random_state=42)
    	X_train.append(a_train)
    	X_test.append(a_test)
    	y_train.append(b_train)
    	y_test.append(b_test)
    
    return X_train,y_train,X_test,y_test,mask
           
def load_satflow_all():
    file_path='csmt/datasets/data/Sat-Flow/Sat_Flow.csv'
    model_file='csmt/datasets/pickles/Sat_Flow_dataframe.pkl'

    if path.exists(model_file):
        df = pd.read_pickle(model_file)
        X = df.drop(['Label'], axis=1)
        y = df['Label']
        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
        return X_train,y_train,X_test,y_test

    label_map = {'BENIGN': 0, 'Syn_DDoS': 1, 'UDP_DDoS': 1, 'Botnet': 1,'Web Attack': 1, 'Backdoor': 1,'LDAP_DDoS': 1, 'MSSQL_DDoS': 1, 'NetBIOS_DDoS': 1, 'Portmap_DDoS': 1}
             
    df = pd.read_csv(file_path, encoding='utf8', low_memory=False)
    ##删除第一列序号，无用信息
    df = df.drop([df.columns[0]], axis=1)
    # 删除字符串左侧空格
    df.columns = df.columns.str.lstrip()
    print(df.shape)
    
    ##删除空值
    df = df.dropna()
    df = df.reset_index(drop=True)
    print(df.shape) 

    ##Label赋值
    df['Label'] = df['Label'].map(label_map)

    # 20%采样
    df = df.sample(frac=0.20, random_state=20)
    print(df.shape)

    print(df['Label'].value_counts())
    
    X = df.drop(['Label'], axis=1)
    mask=get_true_mask([column for column in X])
    y = df['Label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
    df.to_pickle(model_file)
    
    return X_train,y_train,X_test,y_test,mask

def load_satflow():
    file_path='csmt/datasets/data/Sat-Flow/MIX_NET.csv'

    label_map = {'BENIGN': 0, 'Syn_DDoS': 1, 'UDP_DDoS': 1, 'Botnet': 1,'Web Attack': 1, 'Backdoor': 1}

    # 读取csv文件
    df = pd.read_csv(file_path, encoding='utf8', low_memory=False)

    # 删除第一列序号，无用信息
    df = df.drop([df.columns[0]], axis=1)

    # 删除字符串左侧空格
    df.columns = df.columns.str.lstrip()
   
    # 删除空值
    df = df.dropna()
    df = df.reset_index(drop=True)
    
    # 删除LDAP_DDoS、MSSQL_DDoS、NetBIOS_DDoS、Portmap_DDoS
    df = df[~df['Label'].isin(['LDAP_DDoS','MSSQL_DDoS','NetBIOS_DDoS','Portmap_DDoS'])]
    
    # 提取BENIGN
    benign = df[df['Label'] == 'BENIGN']
    benign1 = benign.sample(n = 55000,random_state = 1,axis = 0)  ##采样55000条BENIGN
    benign2 = benign.sample(n = 13000,random_state = 1,axis = 0)  ##采样13000条BENIGN
    benign3 = benign.sample(n = 15000,random_state = 1,axis = 0)  ##采样15000条BENIGN
    
    # ##1.SYN_DDoS+BENIGN
    # syn_ddos = df[df['Label'] == 'Syn_DDoS']
    # #print(syn_ddos.shape)   ##(54789, 31)
    # syn_benign = pd.concat([syn_ddos,benign1]) 
    # syn_benign['Label'] = syn_benign['Label'].map(label_map)
    # X = syn_benign.drop(['Label'], axis=1)
    # y = syn_benign['Label']

    # ##2.UDP_DDoS+BENIGN
    # udp_ddos = df[df['Label'] == 'UDP_DDoS']
    # #print(udp_ddos.shape)        ##(57082, 31)
    # udp_benign = pd.concat([udp_ddos,benign1])
    # udp_benign['Label'] = udp_benign['Label'].map(label_map)
    # X = udp_benign.drop(['Label'], axis=1)
    # y = udp_benign['Label']

    # ##3.Botnet+BENIGN
    # botnet = df[df['Label'] == 'Botnet']
    # #print(botnet.shape)        ##(14622, 31)
    # botnet_benign = pd.concat([botnet,benign3])
    # botnet_benign['Label'] = botnet_benign['Label'].map(label_map)
    # X = botnet_benign.drop(['Label'], axis=1)
    # y = botnet_benign['Label']
    
    # ##4.Web Attack+BENIGN
    # web_attack = df[df['Label'] == 'Web Attack']
    # #print(web_attack.shape)        ##(13017, 31)
    # web_benign = pd.concat([web_attack,benign2])
    # web_benign['Label'] = web_benign['Label'].map(label_map)
    # X = web_benign.drop(['Label'], axis=1)
    # y = web_benign['Label']
    
    # ##5.Backdoor+BENIGN
    backdoor = df[df['Label'] == 'Backdoor']
    # print(backdoor.shape)        ##(12762, 31)
    backdoor_benign = pd.concat([backdoor,benign2])
    backdoor_benign['Label'] = backdoor_benign['Label'].map(label_map)
    X = backdoor_benign.drop(['Label'], axis=1)
    y = backdoor_benign['Label']
    

    print(X.shape)
    print(y.shape)
    mask=get_true_mask([column for column in X])
    return X,y,mask