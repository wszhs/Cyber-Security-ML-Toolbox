
import numpy as np
import pandas as pd
from scapy.all import *

def load_pcap_1():
    # from csmt.datasets._base import get_true_mask
    # import csmt.feature_extractor.AfterImageExtractor.FEKitsune as Fe
    # from csmt.feature_extractor.AfterImageExtractor.KitsuneTools import RunFE 
    # X = np.load('csmt/datasets/data/Kitsune/mirai/kitsune_feature_data.npy')
    # y= np.load('csmt/datasets/data/Kitsune/mirai/labels.npy')
    # print(X.shape)

    # X = np.load('csmt/datasets/data/Kitsune/SYN_DoS/kitsune_feature_data.npy')
    # y= np.load('csmt/datasets/data/Kitsune/SYN_DoS/labels.npy')
    # print(y.shape)

    # X=pd.DataFrame(X)
    # y=pd.DataFrame(y)

    from csmt.datasets._base import get_true_mask
    import csmt.feature_extractor.AfterImageExtractor.FEKitsune as Fe
    from csmt.feature_extractor.AfterImageExtractor.KitsuneTools import RunFE 

    pcap_file='csmt/datasets/data/Kitsune/mirai/Mirai_pcap.pcap'
    label_file='csmt/datasets/data/Kitsune/mirai/mirai_labels.csv'

    # pcap_file='csmt/datasets/data/Kitsune/SYN_DoS/SYN DoS_pcap.pcap'
    # label_file='csmt/datasets/data/Kitsune/SYN_DoS/labels.npy'

    import time
    start_time=time.time()
    scapy_pcap=rdpcap(pcap_file)
    X = Fe.Kitsune(scapy_pcap, np.Inf)
    X, _ = RunFE(X)
    end_time=time.time()
    print('time cost:',end_time-start_time)

    X=np.array(X)
    # np.save("syn_dos.npy",X)
    # y=pd.read_csv(label_file)
    y=y.values
    mask=get_true_mask([column for column in X])
    return X,y,mask

def load_pcap():
    from csmt.datasets._base import get_true_mask
    import csmt.feature_extractor.AfterImageExtractor.FEKitsune as Fe
    from csmt.feature_extractor.AfterImageExtractor.KitsuneTools import RunFE 

    normal_pcap_file = 'tests/example/normal.pcap'
    mal_pcap_file='tests/example/malware.pcap'

    normal_scapy = rdpcap(normal_pcap_file)
    mal_scapy = rdpcap(mal_pcap_file)

    normal_fe = Fe.Kitsune(normal_scapy, np.Inf)
    mal_fe=Fe.Kitsune(mal_scapy, np.Inf)

    normal_feature, _ = RunFE(normal_fe)
    mal_feature, _ = RunFE(mal_fe)

    normal_feature=np.array(normal_feature)
    mal_feature=np.array(mal_feature)

    X=np.concatenate((normal_feature,mal_feature),axis=0)
    y=np.concatenate((np.zeros(normal_feature.shape[0]),np.ones(mal_feature.shape[0])),axis=0)

    X_train,y_train,X_val,y_val,X_test,y_test=X,y,X,y,X,y
    mask=get_true_mask([column for column in X])
 
    return  X_train,y_train,X_val,y_val,X_test,y_test,mask

# nprint
def load_pcap_nprint():
    from csmt.datasets._base import get_true_mask
    normal_pcap_file = 'tests/example/normal.pcap'
    mal_pcap_file='tests/example/malware.pcap'
    # subprocess.run('nprint -P tests/example/malware.pcap -t -W tests/example/malware.npt', shell=True)
    # subprocess.run('nprint -P tests/example/normal.pcap -t -W tests/example/normal.npt', shell=True)

    normal_fe=pd.read_csv('tests/example/normal.npt', index_col=0).values
    mal_fe= pd.read_csv('tests/example/malware.npt', index_col=0).values
    print(normal_fe.shape)
    print(mal_fe.shape)
    X=np.concatenate((normal_fe, mal_fe),axis=0)
    y=np.concatenate((np.zeros(normal_fe.shape[0]),np.ones(mal_fe.shape[0])),axis=0)

    X_train,y_train,X_val,y_val,X_test,y_test=X,y,X,y,X,y
    mask=get_true_mask([column for column in X])

    return  X_train,y_train,X_val,y_val,X_test,y_test,mask

