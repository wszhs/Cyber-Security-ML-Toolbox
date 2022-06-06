'''
Author: your name
Date: 2021-07-16 14:07:33
LastEditTime: 2021-07-16 15:54:49
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /Cyber-Security-ML-Toolbox/csmt/classifiers/anomaly_detection/KitNET/train.py
'''
from model import tKitsune,eKitsune
from model import RunKN
import numpy as np
import pickle as pkl


feat_file_path='csmt/datasets/data/Kitsune/train_ben.npy'
feat_test_file_path='csmt/datasets/data/Kitsune/test.npy'
maxAE=10

print("Warning: under TRAIN mode!")
feature = np.load(feat_file_path)
feature_size = feature.shape[1]
model=[]
tkn = tKitsune(model,feature_size, maxAE, 500, 5000)
rmse = RunKN(tkn, feature)
AD_threshold = max(rmse[500:])
model.append(AD_threshold)

feature_test = np.load(feat_test_file_path)
ekn = eKitsune(model,feature_size, maxAE)
rmse = RunKN(ekn, feature_test)
rmse = np.array(rmse)

AD_threshold = model[3]

print('AD_threshold:', AD_threshold)
print('# rmse over AD_t:', rmse[rmse > AD_threshold].shape)
print('Total number:', len(rmse))
print("rmse mean:", np.mean(rmse))
