'''
Author: your name
Date: 2021-03-25 14:27:29
LastEditTime: 2021-07-03 09:22:22
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /NashHE/nashhe/datasets/_load_cicandmal2017.py
'''
# from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from os import path

# from csmt.datasets._base import get_mask,get_true_mask,add_str,get_dict

def load_cicandmal2017_new(data_class_type):
    if data_class_type=='single_binary' or data_class_type=='multi':
        raise Exception('Data classification type is not supported！')
    file_path='/Users/zhanghangsheng/Documents/my_code/Cyber-Security-ML-Toolbox/csmt/datasets/data/CICAndMal2017/CicFlowMeterData.csv'
    model_file='csmt/datasets/pickles/cicandmal2017_dataframe.pkl'
    
    if path.exists(model_file):
        df = pd.read_pickle(model_file)
        X = df.drop(['Label'], axis=1)
        y = df['Label']
        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
        return X_train,y_train,X_test,y_test

    label_map = {'BENIGN': 0, 'SCAREWARE_FAKEAV': 1, 'SCAREWARE_FAKEAPP': 1, 'SCAREWARE_FAKEAPPAL': 1,
             'SCAREWARE_ANDROIDDEFENDER': 1, 'SCAREWARE_VIRUSSHIELD': 1, 'SCAREWARE_FAKEJOBOFFER': 1, 'MALWARE': 1,
             'SCAREWARE_PENETHO': 1, 'SCAREWARE_FAKETAOBAO': 1, 'SCAREWARE_AVPASS': 1, 'SCAREWARE_ANDROIDSPY': 1,
             'SCAREWARE_AVFORANDROID': 1, 'ADWARE_FEIWO': 1, 'ADWARE_GOOLIGAN': 1, 'ADWARE_KEMOGE': 1,
             'ADWARE_EWIND': 1, 'ADWARE_YOUMI': 1, 'ADWARE_DOWGIN': 1, 'ADWARE_SELFMITE': 1, 'ADWARE_KOODOUS': 1,
             'ADWARE_MOBIDASH': 1, 'ADWARE_SHUANET': 1, 'SMSMALWARE_FAKEMART': 1, 'SMSMALWARE_ZSONE': 1,
             'SMSMALWARE_FAKEINST': 1, 'SMSMALWARE_MAZARBOT': 1, 'SMSMALWARE_NANDROBOX': 1, 'SMSMALWARE_JIFAKE': 1,
             'SMSMALWARE_SMSSNIFFER': 1, 'SMSMALWARE_BEANBOT': 1, 'SCAREWARE': 1, 'SMSMALWARE_FAKENOTIFY': 1,
             'SMSMALWARE_PLANKTON': 1, 'SMSMALWARE_BIIGE': 1, 'RANSOMWARE_LOCKERPIN': 1, 'RANSOMWARE_CHARGER': 1,
             'RANSOMWARE_PORNDROID': 1, 'RANSOMWARE_PLETOR': 1, 'RANSOMWARE_JISUT': 1, 'RANSOMWARE_WANNALOCKER': 1,
             'RANSOMWARE_KOLER': 1, 'RANSOMWARE_RANSOMBO': 1, 'RANSOMWARE_SIMPLOCKER': 1, 'RANSOMWARE_SVPENG': 1}
             
    df = pd.read_csv(file_path, encoding='utf8', low_memory=False)
    # the FLOW ID column seems a combination of source and destination IP, port and protocal
    # hence useless to me
    df = df.drop([df.columns[0]], axis=1)
    # remove white space at the beginning of string in dataframe header
    df.columns = df.columns.str.lstrip()

    print(df.shape)
    df = df.dropna()
    df = df.reset_index(drop=True)
    print(df.shape) 

    df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
    df['Packet Length Std'] = pd.to_numeric(df['Packet Length Std'], errors='coerce')
    df['CWE Flag Count'] = pd.to_numeric(df['CWE Flag Count'], errors='coerce')
    df['Active Mean'] = pd.to_numeric(df['Active Mean'], errors='coerce')
    df['Flow IAT Mean'] = pd.to_numeric(df['Flow IAT Mean'], errors='coerce')
    df['URG Flag Count'] = pd.to_numeric(df['URG Flag Count'], errors='coerce')
    df['Down/Up Ratio'] = pd.to_numeric(df['Down/Up Ratio'], errors='coerce')
    df['Fwd Avg Bytes/Bulk'] = pd.to_numeric(df['Fwd Avg Bytes/Bulk'], errors='coerce')
    df['Flow IAT Min'] = pd.to_numeric(df['Flow IAT Min'], errors='coerce')
    df['Source IP'] = df['Source IP'].astype('str')
    df['Destination IP'] = df['Destination IP'].astype('str')
    df['Label'] = df['Label'].map(label_map)

    # Drop the features which have only 1 unique value:
    nunique = df.apply(pd.Series.nunique)
    cols_to_drop = nunique[nunique == 1].index
    df.drop(cols_to_drop, axis=1, inplace=True)
    print(df.shape)

    # sample 20%
    df = df.sample(frac=0.01, random_state=20)

    time_df = df['Timestamp'].value_counts()
    x = pd.date_range('2017-01-01', '2017-12-31', freq='D').astype(str)
    time_map = dict(zip(x, np.arange(1, 366)))
    df['Timestamp'] = df['Timestamp'].astype(str).str[:10].map(time_map)
    
    # Label Encoding
    df["Source IP"] = LabelEncoder().fit_transform(df["Source IP"])
    df["Destination IP"] = LabelEncoder().fit_transform(df["Destination IP"])


    feature_array=['Total Length of Bwd Packets','Fwd Packet Length Std','Bwd Packet Length Min','Bwd Packet Length Std','Flow IAT Max','Flow IAT Min','Init_Win_bytes_forward','Init_Win_bytes_backward','min_seg_size_forward']
    # X = df.drop(['Label','label','Timestamp'], axis=1)
    X=df[feature_array]
    print(X)
    y = df['Label']

    print(X[0:10])
    return X,y,mask

def load_cicandmal2017():
    from csmt.datasets._base import get_mask,get_true_mask
    file_path='csmt/datasets/data/CICAndMal2017-log/cicandmal2017_all.csv'
    model_file='csmt/datasets/pickles/cicandmal2017_dataframe.pkl'

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

    allow_array=independent_feature_group2+middle_group4
    # feature_array = [i for i in feature_array if i not in dependent_feature_group3]
    feature_array=feature_array

    label_map={'Benign':0,'blige':1,'davforandroid':1,'davpass':1,'dbeanbot':1,'dcharger':1,'dowgin':1,'koler':1,'youmi':1}
    if path.exists(model_file):
        df = pd.read_pickle(model_file)
        X = df.drop(['label'], axis=1)
        mask=get_mask([column for column in X],allow_array)
        y = df['label']
        return X,y,mask

    df = pd.read_csv(file_path, encoding='utf8', low_memory=False)

    df=df[feature_array]
    # print(df['label'].value_counts())
    # remove white space at the beginning of string in dataframe header
    df.columns = df.columns.str.lstrip()

    df['label'] = df['label'].map(label_map)

    # sample 80%
    df = df.sample(frac=0.10, random_state=20)
    df.to_pickle(model_file)
    print(df['label'].value_counts())
    # print(df.dtypes)

    X = df.drop(['label'], axis=1)
    mask=get_mask([column for column in X],allow_array)
    y = df['label']
    return X,y,mask

def load_cicandmal2017_spl(data_class_type):
    if data_class_type=='single_binary' or data_class_type=='multi':
        raise Exception('Data classification type is not supported！')
    file_path='csmt/datasets/data/CICAndMal2017-log/cicandmal2017_all.csv'
    model_file='csmt/datasets/pickles/cicandmal2017_spl_dataframe.pkl'

    feature_array=['orig_spl','resp_spl','orig_spt','resp_spt','label']

    label_map={'Benign':0,'blige':1,'davforandroid':1,'davpass':1,'dbeanbot':1,'dcharger':1,'dowgin':1,'koler':1,'youmi':1}

    if path.exists(model_file):
        df = pd.read_pickle(model_file)
        X = df.drop(['label'], axis=1)
        mask=get_true_mask([column for column in X])
        y = df['label']
        return X,y,mask

    df = pd.read_csv(file_path, encoding='utf8', low_memory=False)

    df=df[feature_array]

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

    df = df.sample(frac=0.10, random_state=20)
    df['label'] = df['label'].map(label_map)
    print(df['label'].value_counts())
    # print(df.dtypes)

    X = df.drop(['orig_spl','resp_spl','orig_spt','resp_spt','label'], axis=1)
    mask=get_true_mask([column for column in X])
    y = df['label']
    return X,y,mask

def load_cicandmal2017_tls(data_class_type):

    if data_class_type=='single_binary' or data_class_type=='multi':
        raise Exception('Data classification type is not supported！')
    file_path='csmt/datasets/data/CICAndMal2017-log/cicandmal2017_all.csv'
    model_file='csmt/datasets/pickles/cicandmal2017_tls_dataframe.pkl'

    feature_array=['client_ciphers','client_curves','ssl_server_exts','ssl_client_exts','cipher','label']

    label_map={'Benign':0,'blige':1,'davforandroid':1,'davpass':1,'dbeanbot':1,'dcharger':1,'dowgin':1,'koler':1,'youmi':1}

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

        # sample 80%
    df = df.sample(frac=0.10, random_state=20)

    df['label'] = df['label'].map(label_map)

    print(df['label'].value_counts())
    X = df.drop(['label'], axis=1)
    mask=get_true_mask([column for column in X])
    y = df['label']
    # print(X.isnull().sum())
    return X,y,mask

def load_cicandmal2017_all(data_class_type):
    if data_class_type=='single_binary' or data_class_type=='multi':
        raise Exception('Data classification type is not supported！')
    file_path='csmt/datasets/data/CicAndMal2017-log/cicandmal2017_all.csv'

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
                    'bwd_init_window_size','fwd_last_window_size','bwd_last_window_size','orig_spl','resp_spl','orig_spt','resp_spt','client_ciphers','client_curves','ssl_server_exts','ssl_client_exts','cipher','label']

    unallowed_feature_group1=['bwd_pkts_tot','fwd_data_pkts_tot','bwd_data_pkts_tot','bwd_pkts_payload.max','bwd_pkts_payload.min','bwd_pkts_payload.avg',
                            'bwd_pkts_payload.std','bwd_iat.tot','bwd_iat.avg','bwd_iat.std','bwd_iat.max','bwd_iat.min','bwd_PSH_flag_count','bwd_URG_flag_count',
                            'bwd_header_size_tot','bwd_pkts_per_sec','bwd_bulk_bytes','bwd_bulk_packets','bwd_bulk_rate','fwd_subflow_pkts','fwd_subflow_bytes','bwd_subflow_pkts',
                            'bwd_subflow_bytes','bwd_init_window_size']
    independent_feature_group2=['flow_duration','fwd_pkts_tot','fwd_pkts_payload.max','fwd_pkts_payload.min','flow_iat.max','flow_iat.min',
                                'fwd_iat.max','fwd_iat.min','fwd_PSH_flag_count','fwd_URG_flag_count','flow_FIN_flag_count','flow_SYN_flag_count','flow_RST_flag_count',
                                'flow_ACK_flag_count','flow_CWR_flag_count','flow_ECE_flag_count','fwd_init_window_size','client_ciphers','client_curves','ssl_server_exts','ssl_client_exts','cipher']
    dependent_feature_group3=['fwd_pkts_payload.avg','payload_bytes_per_second','flow_pkts_per_sec','flow_iat.avg','fwd_iat.tot','fwd_iat.avg','fwd_header_size_tot',
                                'fwd_pkts_per_sec','flow_pkts_payload.min','flow_pkts_payload.max','flow_pkts_payload.avg','down_up_ratio','act_data_pkt_fwd']
    middle_group4=['fwd_pkts_payload.std','flow_iat.std','fwd_iat.std','flow_pkts_payload.std','fwd_bulk_bytes','fwd_bulk_packets','fwd_bulk_rate',
                    'min_seg_size_forward','active.avg','active.std','active.max','active.min','idle.avg','idle.std','idle.max','idle.min']

    allow_array=independent_feature_group2+middle_group4
    feature_array = [i for i in feature_array if i not in dependent_feature_group3]


    label_map={'Benign':0,'blige':1,'davforandroid':1,'davpass':1,'dbeanbot':1,'dcharger':1,'dowgin':1,'koler':1,'youmi':1}

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
