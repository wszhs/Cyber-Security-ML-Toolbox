'''
Author: your name
Date: 2021-07-15 15:03:35
LastEditTime: 2021-07-20 19:44:31
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /Cyber-Security-ML-Toolbox/csmt/classifiers/anomaly_detection/diff_rf.py
'''
from csmt.classifiers.abstract_model import AbstractModel
class DIFFRF(AbstractModel):

    def __init__(self,input_size,output_size):
        from csmt.classifiers.anomaly_detection.diff_rf_packet.diff_RF import DiFF_TreeEnsemble
        from csmt.estimators.classification.anomaly_classifier_diff import AnomalyClassifeirDIFF
        model=DiFF_TreeEnsemble(n_trees=256)
        self.classifier=AnomalyClassifeirDIFF(model=model,nb_features=input_size, nb_classes=output_size,clip_values=(0,1),contamination=0.1)