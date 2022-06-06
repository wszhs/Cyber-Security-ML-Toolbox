
import csmt.feature_extractor.AfterImageExtractor.FEKitsune as Fe
from csmt.feature_extractor.AfterImageExtractor.KitsuneTools import RunFE 
from csmt.normalizer import Normalizer
import numpy as np

def get_norm(X):
    normer = Normalizer(X.shape[-1],online_minmax=False)
    X = normer.fit_transform(X)
    return normer

def get_feature(pcap):
    local_FE=Fe.Kitsune(pcap, np.Inf)
    feature, _ = RunFE(local_FE)
    feature=np.array(feature)
    return feature

def get_new_feature(pcap,inter_arr):
    mal_pos = []
    cft_num = 0
    grp_size=inter_arr.mal.shape[0]
    for i in range(grp_size):
        cft_num += int(round(inter_arr.mal[i][1]))
        mal_pos.append(i + cft_num)
    local_FE = Fe.Kitsune(pcap, np.Inf, True)
    feature, all_feature = RunFE(local_FE,origin_pos=mal_pos)
    return feature