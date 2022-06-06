'''
Author: your name
Date: 2021-04-23 14:38:01
LastEditTime: 2021-07-10 19:10:47
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /Cyber-Security-ML-Toolbox/csmt/datasets/_load_ctu13.py
'''
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
import pickle
import pandas as pd
import numpy as np
from os import path
from csmt.datasets._base import get_mask,get_true_mask,add_str,get_dict

def load_datacon():

    file_path='csmt/datasets/data/datacon-log/datacon_flowmeter.csv'

    feature_array=['flow_duration','fwd_pkts_tot','bwd_pkts_tot','fwd_data_pkts_tot','bwd_data_pkts_tot',
                    'fwd_pkts_per_sec','bwd_pkts_per_sec','flow_pkts_per_sec','down_up_ratio','fwd_header_size_tot',
                    'fwd_header_size_min','fwd_header_size_max','bwd_header_size_tot','bwd_header_size_min',
                    'bwd_header_size_max','flow_FIN_flag_count','flow_SYN_flag_count','flow_RST_flag_count',
                    'fwd_PSH_flag_count','bwd_PSH_flag_count','flow_ACK_flag_count','fwd_URG_flag_count',
                    'bwd_URG_flag_count','flow_CWR_flag_count','flow_ECE_flag_count','fwd_pkts_payload.min',
                    'fwd_pkts_payload.max','fwd_pkts_payload.tot','fwd_pkts_payload.avg','fwd_pkts_payload.std',
                    'bwd_pkts_payload.min','bwd_pkts_payload.max','bwd_pkts_payload.tot','bwd_pkts_payload.avg',
                    'bwd_pkts_payload.std','flow_pkts_payload.min','flow_pkts_payload.max','flow_pkts_payload.tot',
                    'flow_pkts_payload.avg','flow_pkts_payload.std','fwd_iat.min','fwd_iat.max','fwd_iat.tot','fwd_iat.avg',
                    'fwd_iat.std','bwd_iat.min','bwd_iat.max','bwd_iat.tot','bwd_iat.avg','bwd_iat.std','flow_iat.min',
                    'flow_iat.max','flow_iat.tot','flow_iat.avg','flow_iat.std','payload_bytes_per_second','fwd_subflow_pkts',
                    'bwd_subflow_pkts','fwd_subflow_bytes','bwd_subflow_bytes','fwd_bulk_bytes','bwd_bulk_bytes',
                    'fwd_bulk_packets','bwd_bulk_packets','fwd_bulk_rate','bwd_bulk_rate','active.min','active.max','active.tot',
                    'active.avg','active.std','idle.min','idle.max','idle.tot','idle.avg','idle.std','fwd_init_window_size',
                    'bwd_init_window_size','fwd_last_window_size','bwd_last_window_size','label']

    unallowed_feature_group1=['bwd_pkts_tot','fwd_data_pkts_tot','bwd_data_pkts_tot','bwd_pkts_payload.max','bwd_pkts_payload.min','bwd_pkts_payload.avg',
                            'bwd_pkts_payload.std','bwd_iat.tot','bwd_iat.avg','bwd_iat.std','bwd_iat.max','bwd_iat.min','bwd_PSH_flag_count','bwd_URG_flag_count',
                            'bwd_header_size_tot','bwd_pkts_per_sec','bwd_bulk_bytes','bwd_bulk_packets','bwd_bulk_rate','fwd_subflow_pkts','fwd_subflow_bytes','bwd_subflow_pkts',
                            'bwd_subflow_bytes','bwd_init_window_size']
    independent_feature_group2=['flow_duration','fwd_pkts_tot','fwd_pkts_payload.max','fwd_pkts_payload.min','flow_iat.max','flow_iat.min',
                                'fwd_iat.max','fwd_iat.min','fwd_PSH_flag_count','fwd_URG_flag_count','flow_FIN_flag_count','flow_SYN_flag_count','flow_RST_flag_count',
                                'flow_ACK_flag_count','flow_CWR_flag_count','flow_ECE_flag_count','fwd_init_window_size']
    dependent_feature_group3=['fwd_pkts_payload.avg','payload_bytes_per_second','flow_pkts_per_sec','flow_iat.avg','fwd_iat.tot','fwd_iat.avg','fwd_header_size_tot',
                                'fwd_pkts_per_sec','flow_pkts_payload.min','flow_pkts_payload.max','flow_pkts_payload.avg','down_up_ratio','act_data_pkt_fwd']
    middle_group4=['fwd_pkts_payload.std','flow_iat.std','fwd_iat.std','flow_pkts_payload.std','fwd_bulk_bytes','fwd_bulk_packets','fwd_bulk_rate',
                    'min_seg_size_forward','active.avg','active.std','active.max','active.min','idle.avg','idle.std','idle.max','idle.min']

    allow_array=independent_feature_group2
    feature_array = [i for i in feature_array if i not in dependent_feature_group3]

    label_map={'white':0,'black':1}

    df = pd.read_csv(file_path, encoding='utf8', low_memory=False)

    df=df[feature_array]
    print(df['label'].value_counts())
    # remove white space at the beginning of string in dataframe header
    df.columns = df.columns.str.lstrip()

    df['label'] = df['label'].map(label_map)
    print(df.dtypes)

    # sample 80%
    df_0=df[df['label']==0]
    df_0 = df_0.sample(frac=0.2, random_state=20)
    df_1=df[df['label']==1]
    df_1 = df_1.sample(frac=0.1, random_state=20)
    df= pd.concat([df_0,df_1])
    print(df['label'].value_counts())

    X = df.drop(['label'], axis=1)

    y = df['label']
    print(y.value_counts())

    mask=get_mask([column for column in X],allow_array)
    return X,y,mask

def load_datacon_spl(data_class_type):
    if data_class_type=='single_binary' or data_class_type=='multi':
        raise Exception('Data classification type is not supported！')
    file_path='csmt/datasets/data/datacon-log/datacon_spl.csv'
    model_file='csmt/datasets/pickles/datacon_spl_dataframe.pkl'

    feature_array=['orig_spl','resp_spl','orig_spt','resp_spt','label']

    label_map={'white':0,'black':1}

    if path.exists(model_file):
        df = pd.read_pickle(model_file)
        X = df.drop(['label'], axis=1)
        mask=get_true_mask([column for column in X])
        y = df['label']
        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
        return X_train,y_train,X_test,y_test

    df = pd.read_csv(file_path, encoding='utf8', low_memory=False)

    df=df[feature_array]
    print(df['label'].value_counts())

    df['orig_spl']=df['orig_spl'].map(lambda x:add_str(str(x)))
    df['resp_spl']=df['resp_spl'].map(lambda x:add_str(str(x)))
    df['orig_spt']=df['orig_spt'].map(lambda x:add_str(str(x)))
    df['resp_spt']=df['resp_spt'].map(lambda x:add_str(str(x)))

    for i in range(20):
        df['orig_spl_'+str(i)]=df['orig_spl'].map(lambda x:str(x).split(',')[i])
        df['orig_spl_'+str(i)] = pd.to_numeric(df['orig_spl_'+str(i)], errors='coerce')
        df.drop(df[np.isnan(df['orig_spl_'+str(i)])].index, inplace=True)

        df['resp_spl_'+str(i)]=df['resp_spl'].map(lambda x:str(x).split(',')[i])
        df['resp_spl_'+str(i)] = pd.to_numeric(df['resp_spl_'+str(i)], errors='coerce')
        df.drop(df[np.isnan(df['resp_spl_'+str(i)])].index, inplace=True)

        df['orig_spt_'+str(i)]=df['orig_spt'].map(lambda x:str(x).split(',')[i])
        df['orig_spt_'+str(i)] = pd.to_numeric(df['orig_spt_'+str(i)], errors='coerce')
        df.drop(df[np.isnan(df['orig_spt_'+str(i)])].index, inplace=True)

        df['resp_spt_'+str(i)]=df['resp_spt'].map(lambda x:str(x).split(',')[i])
        df['resp_spt_'+str(i)] = pd.to_numeric(df['resp_spt_'+str(i)], errors='coerce')
        df.drop(df[np.isnan(df['resp_spt_'+str(i)])].index, inplace=True)

    df['label'] = df['label'].map(label_map)

    df_0=df[df['label']==0]
    df_0 = df_0.sample(frac=0.2, random_state=20)
    df_1=df[df['label']==1]
    df_1 = df_1.sample(frac=0.4, random_state=20)
    df= pd.concat([df_0,df_1])

    print(df['label'].value_counts())
    # print(df.dtypes)

    X = df.drop(['orig_spl','resp_spl','orig_spt','resp_spt','label'], axis=1)
    mask=get_true_mask([column for column in X])
    y = df['label']
    return X,y,mask

def load_datacon_tls(data_class_type):

    if data_class_type=='single_binary' or data_class_type=='multi':
        raise Exception('Data classification type is not supported！')
    file_path='csmt/datasets/data/datacon-log/datacon_all.csv'
    model_file='csmt/datasets/pickles/datacon_tls_dataframe.pkl'

    feature_array=['client_ciphers','client_curves','ssl_server_exts','ssl_client_exts','cipher','label']

    label_map={'white':0,'black':1}

    df = pd.read_csv(file_path, encoding='utf8', low_memory=False)

    df=df[feature_array]
    print(df['label'].value_counts())

    dict_val=df['client_ciphers'].value_counts().items()
    cc_map=get_dict(dict_val)
    df['client_ciphers']=df['client_ciphers'].map(cc_map)

    dict_val=df['client_curves'].value_counts().items()
    cc_map=get_dict(dict_val)
    df['client_curves']=df['client_curves'].map(cc_map)

    dict_val=df['ssl_server_exts'].value_counts().items()
    cc_map=get_dict(dict_val)
    df['ssl_server_exts']=df['ssl_server_exts'].map(cc_map)

    dict_val=df['ssl_client_exts'].value_counts().items()
    cc_map=get_dict(dict_val)
    df['ssl_client_exts']=df['ssl_client_exts'].map(cc_map)

    dict_val=df['cipher'].value_counts().items()
    cc_map=get_dict(dict_val)
    df['cipher']=df['cipher'].map(cc_map)

    df.drop(df[np.isnan(df['client_ciphers'])].index, inplace=True)
    df.drop(df[np.isnan(df['client_curves'])].index, inplace=True)
    df.drop(df[np.isnan(df['ssl_server_exts'])].index, inplace=True)
    df.drop(df[np.isnan(df['ssl_client_exts'])].index, inplace=True)
    df.drop(df[np.isnan(df['cipher'])].index, inplace=True)

    df['label'] = df['label'].map(label_map)
    
    print(df['label'].value_counts())

    X = df.drop(['label'], axis=1)
    mask=get_true_mask([column for column in X])
    y = df['label']
    # print(X.isnull().sum())
    return X,y,mask

def load_datacon_all(data_class_type):
    if data_class_type=='single_binary' or data_class_type=='multi':
        raise Exception('Data classification type is not supported！')
    file_path='csmt/datasets/data/datacon-log/datacon_all.csv'

    feature_array=['flow_duration','fwd_pkts_tot','bwd_pkts_tot','fwd_data_pkts_tot','bwd_data_pkts_tot',
                    'fwd_pkts_per_sec','bwd_pkts_per_sec','flow_pkts_per_sec','down_up_ratio','fwd_header_size_tot',
                    'fwd_header_size_min','fwd_header_size_max','bwd_header_size_tot','bwd_header_size_min',
                    'bwd_header_size_max','flow_FIN_flag_count','flow_SYN_flag_count','flow_RST_flag_count',
                    'fwd_PSH_flag_count','bwd_PSH_flag_count','flow_ACK_flag_count','fwd_URG_flag_count',
                    'bwd_URG_flag_count','flow_CWR_flag_count','flow_ECE_flag_count','fwd_pkts_payload.min',
                    'fwd_pkts_payload.max','fwd_pkts_payload.tot','fwd_pkts_payload.avg','fwd_pkts_payload.std',
                    'bwd_pkts_payload.min','bwd_pkts_payload.max','bwd_pkts_payload.tot','bwd_pkts_payload.avg',
                    'bwd_pkts_payload.std','flow_pkts_payload.min','flow_pkts_payload.max','flow_pkts_payload.tot',
                    'flow_pkts_payload.avg','flow_pkts_payload.std','fwd_iat.min','fwd_iat.max','fwd_iat.tot','fwd_iat.avg',
                    'fwd_iat.std','bwd_iat.min','bwd_iat.max','bwd_iat.tot','bwd_iat.avg','bwd_iat.std','flow_iat.min',
                    'flow_iat.max','flow_iat.tot','flow_iat.avg','flow_iat.std','payload_bytes_per_second','fwd_subflow_pkts',
                    'bwd_subflow_pkts','fwd_subflow_bytes','bwd_subflow_bytes','fwd_bulk_bytes','bwd_bulk_bytes',
                    'fwd_bulk_packets','bwd_bulk_packets','fwd_bulk_rate','bwd_bulk_rate','active.min','active.max','active.tot',
                    'active.avg','active.std','idle.min','idle.max','idle.tot','idle.avg','idle.std','fwd_init_window_size',
                    'bwd_init_window_size','fwd_last_window_size','bwd_last_window_size','orig_spl','resp_spl','orig_spt','resp_spt','client_ciphers','cipher','label']
                    
    unallowed_feature_group1=['bwd_pkts_tot','fwd_data_pkts_tot','bwd_data_pkts_tot','bwd_pkts_payload.max','bwd_pkts_payload.min','bwd_pkts_payload.avg',
                            'bwd_pkts_payload.std','bwd_iat.tot','bwd_iat.avg','bwd_iat.std','bwd_iat.max','bwd_iat.min','bwd_PSH_flag_count','bwd_URG_flag_count',
                            'bwd_header_size_tot','bwd_pkts_per_sec','bwd_bulk_bytes','bwd_bulk_packets','bwd_bulk_rate','fwd_subflow_pkts','fwd_subflow_bytes','bwd_subflow_pkts',
                            'bwd_subflow_bytes','bwd_init_window_size']
    independent_feature_group2=['flow_duration','fwd_pkts_tot','fwd_pkts_payload.max','fwd_pkts_payload.min','flow_iat.max','flow_iat.min',
                                'fwd_iat.max','fwd_iat.min','fwd_PSH_flag_count','fwd_URG_flag_count','flow_FIN_flag_count','flow_SYN_flag_count','flow_RST_flag_count',
                                'flow_ACK_flag_count','flow_CWR_flag_count','flow_ECE_flag_count','fwd_init_window_size','client_ciphers','cipher']
    dependent_feature_group3=['fwd_pkts_payload.avg','payload_bytes_per_second','flow_pkts_per_sec','flow_iat.avg','fwd_iat.tot','fwd_iat.avg','fwd_header_size_tot',
                                'fwd_pkts_per_sec','flow_pkts_payload.min','flow_pkts_payload.max','flow_pkts_payload.avg','down_up_ratio','act_data_pkt_fwd']
    middle_group4=['fwd_pkts_payload.std','flow_iat.std','fwd_iat.std','flow_pkts_payload.std','fwd_bulk_bytes','fwd_bulk_packets','fwd_bulk_rate',
                    'min_seg_size_forward','active.avg','active.std','active.max','active.min','idle.avg','idle.std','idle.max','idle.min']

    allow_array=independent_feature_group2
    feature_array = [i for i in feature_array if i not in dependent_feature_group3]


    label_map={'white':0,'black':1}

    df = pd.read_csv(file_path, encoding='utf8', low_memory=False)

    df=df[feature_array]

    df['orig_spl']=df['orig_spl'].map(lambda x:add_str(str(x)))
    df['resp_spl']=df['resp_spl'].map(lambda x:add_str(str(x)))
    df['orig_spt']=df['orig_spt'].map(lambda x:add_str(str(x)))
    df['resp_spt']=df['resp_spt'].map(lambda x:add_str(str(x)))

    spl_allow=[]

    for i in range(20):
        df['orig_spl_'+str(i)]=df['orig_spl'].map(lambda x:str(x).split(',')[i])
        df['orig_spl_'+str(i)] = pd.to_numeric(df['orig_spl_'+str(i)], errors='coerce')
        df.drop(df[np.isnan(df['orig_spl_'+str(i)])].index, inplace=True)

        df['resp_spl_'+str(i)]=df['resp_spl'].map(lambda x:str(x).split(',')[i])
        df['resp_spl_'+str(i)] = pd.to_numeric(df['resp_spl_'+str(i)], errors='coerce')
        df.drop(df[np.isnan(df['resp_spl_'+str(i)])].index, inplace=True)

        df['orig_spt_'+str(i)]=df['orig_spt'].map(lambda x:str(x).split(',')[i])
        df['orig_spt_'+str(i)] = pd.to_numeric(df['orig_spt_'+str(i)], errors='coerce')
        df.drop(df[np.isnan(df['orig_spt_'+str(i)])].index, inplace=True)

        df['resp_spt_'+str(i)]=df['resp_spt'].map(lambda x:str(x).split(',')[i])
        df['resp_spt_'+str(i)] = pd.to_numeric(df['resp_spt_'+str(i)], errors='coerce')
        df.drop(df[np.isnan(df['resp_spt_'+str(i)])].index, inplace=True)

        spl_allow.append('orig_spl_'+str(i))
        spl_allow.append('resp_spl_'+str(i))
        spl_allow.append('orig_spt_'+str(i))
        spl_allow.append('resp_spt_'+str(i))
    
    allow_array=allow_array+spl_allow

    dict_val=df['client_ciphers'].value_counts().items()
    cc_map=get_dict(dict_val)
    df['client_ciphers']=df['client_ciphers'].map(cc_map)

    dict_val=df['cipher'].value_counts().items()
    cc_map=get_dict(dict_val)
    df['cipher']=df['cipher'].map(cc_map)

    df.drop(df[np.isnan(df['client_ciphers'])].index, inplace=True)
    df.drop(df[np.isnan(df['cipher'])].index, inplace=True)

    print(df['label'].value_counts())
    # remove white space at the beginning of string in dataframe header
    df.columns = df.columns.str.lstrip()

    df['label'] = df['label'].map(label_map)
    print(df.dtypes)

    # sample 10%
    df_0=df[df['label']==0]
    df_0 = df_0.sample(frac=0.1, random_state=20)
    df_1=df[df['label']==1]
    df_1 = df_1.sample(frac=0.1, random_state=20)
    df= pd.concat([df_0,df_1])
    print(df['label'].value_counts())

    X = df.drop(['orig_spl','resp_spl','orig_spt','resp_spt','label'], axis=1)
    y = df['label']
    print(y.value_counts())
    mask=get_mask([column for column in X],allow_array)

    return X,y,mask