
import pandas as pd
import numpy as np
import sys
from os import path
from csmt.datasets._base import get_mask,get_true_mask

def load_cicids2017_han():
    X = np.load('tests/deepaid/timeseries_multi_nids/data/test_feat_cicddos.npy')
    y = np.load('tests/deepaid/timeseries_multi_nids/data/test_label_cicddos.npy')
    X=pd.DataFrame(X)
    y=pd.DataFrame(y)
    df= pd.concat([X,y],axis=1)
    df_1=df[df.iloc[:,-1]==1]
    df_0=df[df.iloc[:,-1]==0]
    # df_0 = df_0.sample(frac=0.99, random_state=20)
    # df_1 = df_1.sample(frac=0.01, random_state=20)

    print('0-'+str(df_0.shape[0]))
    print('1-'+str(df_1.shape[0]))

    df= pd.concat([df_0,df_1])
    X=df.iloc[:,0:-1]
    y=df.iloc[:,-1]
    mask=get_true_mask([column for column in X])

    return X,y,mask

def load_cicids2017_csv():
    ddos_path='csmt/datasets/data/CIC-IDS-2017/MachineLearningCVE/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv'
    normal_path='csmt/datasets/data/CIC-IDS-2017/MachineLearningCVE/Monday-WorkingHours.pcap_ISCX.csv'
    label_map = {'BENIGN': 0,'DDoS': 1}
    df_mal = pd.read_csv(ddos_path, encoding='utf8', low_memory=False)
    # df_normal = pd.read_csv(normal_path, encoding='utf8', low_memory=False)
    
    #正常的样本
    # df_normal = pd.read_csv(normal_path, encoding='utf8', low_memory=False)
    # df_normal.columns = df_normal.columns.str.lstrip()
    # df_normal['Flow Bytes/s'] = pd.to_numeric(df_normal['Flow Bytes/s'], errors='coerce')
    # df_normal['Flow Packets/s'] = pd.to_numeric(df_normal['Flow Packets/s'], errors='coerce')

    # df_normal.replace([np.inf, -np.inf], np.nan, inplace=True)
    # df_normal = df_normal.dropna()
    # df_normal = df_normal.reset_index(drop=True)

    # df_no_label = df_normal.drop(['Label'], axis=1)
    # nunique = df_no_label.apply(pd.Series.nunique)
    # # print(nunique)
    # cols_to_drop = nunique[nunique == 1].index
    # df_normal.drop(cols_to_drop, axis=1, inplace=True)

    # df_normal['Label'] = df_normal['Label'].map(lambda x: 0 if x == "BENIGN" else x)
    # df_0=df_normal[df_normal['Label']==0].iloc[0:2000]

    df_mal.columns = df_mal.columns.str.lstrip()
    df_mal['Flow Bytes/s'] = pd.to_numeric(df_mal['Flow Bytes/s'], errors='coerce')
    df_mal['Flow Packets/s'] = pd.to_numeric(df_mal['Flow Packets/s'], errors='coerce')

    df_mal.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_mal = df_mal.dropna()
    df_mal = df_mal.reset_index(drop=True)

    nunique = df_mal.apply(pd.Series.nunique)
    cols_to_drop = nunique[nunique == 1].index
    df_mal.drop(cols_to_drop, axis=1, inplace=True)

    df_mal['Label'] = df_mal['Label'].map(lambda x: 0 if x == "BENIGN" else x)
    df_mal['Label'] = df_mal['Label'].map(lambda x: 1 if x == "DDoS" else x)

    df_1=df_mal[df_mal['Label']==1]
    df_0=df_mal[df_mal['Label']==0]
    df_1=df_1.iloc[0:2000]
    df_0=df_0.iloc[0:2000]

    print('0-'+str(df_0.shape[0]))
    print('1-'+str(df_1.shape[0]))
    
    df= pd.concat([df_0,df_1])

    df['Label']=df['Label'].astype("int")
    X=df.iloc[:,0:10]
    y=df.iloc[:,-1]
    mask=mask=get_true_mask([column for column in X])

    return X,y,mask


def load_cicids2017_csv_botnet():
    file_path='csmt/datasets/data/CIC-IDS-2017/MachineLearningCVE/Friday-WorkingHours-Morning.pcap_ISCX.csv'
    normal_file_path='csmt/datasets/data/CIC-IDS-2017/MachineLearningCVE/Monday-WorkingHours.pcap_ISCX.csv'
    label_map = {'BENIGN': 0,'Bot': 1}

    #正常的样本
    # df_normal = pd.read_csv(normal_file_path, encoding='utf8', low_memory=False)
    # df_normal.columns = df_normal.columns.str.lstrip()
    # df_normal['Flow Bytes/s'] = pd.to_numeric(df_normal['Flow Bytes/s'], errors='coerce')
    # df_normal['Flow Packets/s'] = pd.to_numeric(df_normal['Flow Packets/s'], errors='coerce')

    # df_normal.replace([np.inf, -np.inf], np.nan, inplace=True)
    # df_normal = df_normal.dropna()
    # df_normal = df_normal.reset_index(drop=True)

    # df_no_label = df_normal.drop(['Label'], axis=1)
    # nunique = df_no_label.apply(pd.Series.nunique)
    # # print(nunique)
    # cols_to_drop = nunique[nunique == 1].index
    # df_normal.drop(cols_to_drop, axis=1, inplace=True)

    # df_normal['Label'] = df_normal['Label'].map(lambda x: 0 if x == "BENIGN" else x)
    # # df_normal = df_normal.sample(frac=0.1, random_state=20)
    # df_0=df_normal[df_normal['Label']==0].iloc[0:2000]

    df_mal = pd.read_csv(file_path, encoding='utf8', low_memory=False)
    # df_mal=df_mal.iloc[:,:]
    # print(df_mal)
    df_mal.columns = df_mal.columns.str.lstrip()
    df_mal['Flow Bytes/s'] = pd.to_numeric(df_mal['Flow Bytes/s'], errors='coerce')
    df_mal['Flow Packets/s'] = pd.to_numeric(df_mal['Flow Packets/s'], errors='coerce')

    df_mal.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_mal = df_mal.dropna()
    df_mal = df_mal.reset_index(drop=True)

    nunique = df_mal.apply(pd.Series.nunique)
    cols_to_drop = nunique[nunique == 1].index
    df_mal.drop(cols_to_drop, axis=1, inplace=True)

    df_mal['Label'] = df_mal['Label'].map(lambda x: 0 if x == "BENIGN" else x)
    df_mal['Label'] = df_mal['Label'].map(lambda x: 1 if x == "Bot" else x)

    df_1=df_mal[df_mal['Label']==1]
    df_0=df_mal[df_mal['Label']==0]

    df_1=df_1.iloc[0:2000]
    df_0=df_0.iloc[0:2000]

    print('0-'+str(df_0.shape[0]))
    print('1-'+str(df_1.shape[0]))
    
    df= pd.concat([df_0,df_1])
    df['Label']=df['Label'].astype("int")
    X = df.drop(['Label'], axis=1)
    # X=df.iloc[:,1:3]
    y = df['Label']
    mask=get_true_mask([column for column in X])

    return X,y,mask

def load_cicids2017():
    try:
        file_path='csmt/datasets/data/CIC-IDS-2017/ddos.pkl'
        df = pd.read_pickle(file_path)
    except IOError:
        print ("Error: 没有找到文件或读取文件失败")
    else:
        print (sys._getframe().f_code.co_name.replace('load_','')+" 数据集加载成功")

    df.columns = df.columns.str.lstrip()
    independent_feature_group2=['Flow Duration','Total Fwd Packets','Total Length of Fwd Packets','Fwd Packet Length Max','Fwd Packet Length Min',
                 'Flow IAT Max','Flow IAT Min','Fwd IAT Max','Fwd IAT Min','Fwd PSH Flags','Fwd URG Flags','FIN Flag Count'
                'SYN Flag Count','RST Flag Count','PSH Flag Count','ACK Flag Count','URG Flag Count','CWE Flag Count','ECE Flag Count','Init_Win_bytes_forward'
                ]
    middle_group4=['Fwd Packet Length Std','Flow IAT Std','Fwd IAT Std','Packet Length Std','Packet Length Variance','Fwd Avg Bytes/Bulk','Fwd Avg Packets/Bulk',
                    'Fwd Avg Bulk Rate','Active Mean','Active Std','Active Max','Active Min','Idle Mean','Idle Std','Idle Max','Idle Min'
                    ]
    allow_array=independent_feature_group2+middle_group4
    df['Flow Bytes/s'] = pd.to_numeric(df['Flow Bytes/s'], errors='coerce')
    df['Flow Packets/s'] = pd.to_numeric(df['Flow Packets/s'], errors='coerce')

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df.dropna()
    df = df.reset_index(drop=True)

    nunique = df.apply(pd.Series.nunique)
    cols_to_drop = nunique[nunique == 1].index
    df.drop(cols_to_drop, axis=1, inplace=True)

    df_1=df[df['Label']==1]
    df_0=df[df['Label']==0]
    df_0=df_0.iloc[0:2000]
    df_1=df_1.iloc[0:2000]

    #botnet
    # df_0 = df_0.sample(frac=0.01, random_state=20)
    # df_1 = df_1.sample(frac=0.5, random_state=20)

    print('0-'+str(df_0.shape[0]))
    print('1-'+str(df_1.shape[0]))
    
    df= pd.concat([df_0,df_1])

    df['Label']=df['Label'].astype("int")
    X=df.iloc[:,0:10]
    y=df.iloc[:,-1]
    # X = df.drop(['Label'], axis=1)
    # y = df['Label']
    # print(X.shape)
    # mask=get_true_mask([column for column in X])
    mask=get_mask([column for column in X],allow_array)

    return X,y,mask
    

def load_cicids2017_old():
    # if data_class_type=='single_binary' or data_class_type=='multi':
    #     raise Exception('Data classification type is not supported！')
    file_path='csmt/datasets/data/CIC-IDS-2017/CicFlowMeterData.csv'
    model_file='csmt/datasets/pickles/cicids2017_dataframe.pkl'
    
    if path.exists(model_file):
        df = pd.read_pickle(model_file)
        X = df.drop(['Label'], axis=1)
        y = df['Label']
        mask=get_true_mask([column for column in X])
        return X,y,mask

    # label_map = {'BENIGN': 0, 'PortScan': 1, 'FTP-Patator': 1, 'SSH-Patator': 1, 'Bot': 1, 'Infiltration': 1,
    #          'Web Attack � Brute Force': 1, 'Web Attack � XSS': 1, 'Web Attack � Sql Injection': 1, 'DDoS': 1,
    #          'DoS slowloris': 1, 'DoS Slowhttptest': 1, 'DoS Hulk': 1, 'DoS GoldenEye': 1, 'Heartbleed': 1}
             
    df = pd.read_csv(file_path, encoding='utf8', low_memory=False)
    df.columns = df.columns.str.lstrip()

    print(df.shape)  # (2830743, 79)
    
    df['Flow Bytes/s'] = pd.to_numeric(df['Flow Bytes/s'], errors='coerce')
    df['Flow Packets/s'] = pd.to_numeric(df['Flow Packets/s'], errors='coerce')

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df.dropna()
    df = df.reset_index(drop=True)
    print(df.shape)  # (2827876, 79)

    # print(list(df.Label.unique()))
    df['Label'] = df['Label'].map(lambda x: 0 if x == "BENIGN" else 1)

    # Drop the features which have only 1 unique value:
    nunique = df.apply(pd.Series.nunique)
    cols_to_drop = nunique[nunique == 1].index
    df.drop(cols_to_drop, axis=1, inplace=True)
    print(df.shape)  # (2827876, 71)

    # sample 20%
    df = df.sample(frac=0.01, random_state=20)
    print(df.shape)
    
    X = df.drop(['Label'], axis=1)
    y = df['Label']
    df.to_pickle(model_file)
    mask=get_true_mask([column for column in X])
    
    return X,y,mask