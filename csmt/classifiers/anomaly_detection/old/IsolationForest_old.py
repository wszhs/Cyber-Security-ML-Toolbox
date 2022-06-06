'''
Author: your name
Date: 2021-07-15 14:54:43
LastEditTime: 2021-07-20 19:18:14
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /Cyber-Security-ML-Toolbox/csmt/classifiers/anomaly_detection/IsolationForest.py
'''
from csmt.classifiers.abstract_model import AbstractModel
from sklearn import ensemble
import numpy as np
from csmt.estimators.classification.anomaly_classifier import AnomalyClassifeir
class IsolationForest(AbstractModel):

    def __init__(self,input_size,output_size):
        model=ensemble.IsolationForest(n_estimators=100, max_samples='auto', contamination=0.1,max_features=1.,bootstrap=False,n_jobs=1, random_state=42)
        self.classifier=AnomalyClassifeir(model=model,nb_features=input_size, nb_classes=output_size,clip_values=(0,1))