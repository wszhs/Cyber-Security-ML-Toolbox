'''
Author: your name
Date: 2021-07-15 14:54:43
LastEditTime: 2021-07-20 19:18:14
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /Cyber-Security-ML-Toolbox/csmt/classifiers/anomaly_detection/IsolationForest.py
'''
from csmt.classifiers.abstract_model import AbstractModel

class IsolationForest(AbstractModel):

    def __init__(self,input_size,output_size):
        from csmt.estimators.classification.anomaly_classifier_if import AnomalyClassifeirIF
        from csmt.classifiers.anomaly_detection.pyod.models.iforest import IForest
        model=IForest()
        self.classifier=AnomalyClassifeirIF(model=model,nb_features=input_size, nb_classes=output_size,clip_values=(0,1),contamination=0.1)