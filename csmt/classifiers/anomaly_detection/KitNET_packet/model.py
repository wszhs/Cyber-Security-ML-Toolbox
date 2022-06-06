'''
Author: your name
Date: 2021-07-15 14:44:56
LastEditTime: 2021-08-05 09:59:43
LastEditors: your name
Description: In User Settings Edit
FilePath: /Cyber-Security-ML-Toolbox/csmt/classifiers/anomaly_detection/KitNET_packet/model.py
'''
'''
Author: your name
Date: 2021-07-15 14:44:56
LastEditTime: 2021-07-16 15:54:37
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /Cyber-Security-ML-Toolbox/csmt/classifiers/anomaly_detection/KitNET_packet/model.py
'''
import sys
sys.path.append("/Users/zhanghangsheng/Documents/my_code/Cyber-Security-ML-Toolbox")
import numpy as np
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
import pickle as pkl
import csmt.classifiers.anomaly_detection.KitNET_packet.executeKitNET.KitNET as eKN
import csmt.classifiers.anomaly_detection.KitNET_packet.trainKitNET.KitNET as tKN
import argparse

def RunKN(K,Feature):
    RMSEs = []
    for i,x in enumerate(Feature):
        rmse = K.proc_next_packet(x)
        # if i%100==0:
        #     print("--- RunKitNET: Pkts",i,"---")
        if rmse == -1:
            break
        RMSEs.append(rmse)
    return RMSEs

def test_mut(mut_feat,model_save_path):

    Feature_Size = mut_feat.shape[1]
    ekn = eKitsune(model_save_path, Feature_Size, 10)
    rmse = RunKN(ekn, mut_feat)

    return rmse

class eKitsune:
    
    def __init__(self,model,feature_size,max_autoencoder_size=10,learning_rate=0.1,hidden_ratio=0.75,):
        
        self.AnomDetector = eKN.KitNET(model,feature_size,max_autoencoder_size,learning_rate,hidden_ratio)

    def proc_next_packet(self,x):
        
        return self.AnomDetector.process(x)  # will train during the grace periods, then execute on all the rest.

class tKitsune:
    def __init__(self,model,n,max_autoencoder_size=10,FM_grace_period=None,AD_grace_period=10000,learning_rate=0.1,hidden_ratio=0.75,):

        self.AnomDetector = tKN.KitNET(model,n,max_autoencoder_size,FM_grace_period,AD_grace_period,learning_rate,hidden_ratio)

    def proc_next_packet(self,x):

        return self.AnomDetector.process(x)  # will train during the grace periods, then execute on all the rest.

    

