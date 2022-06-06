'''
Author: your name
Date: 2021-04-06 14:32:29
LastEditTime: 2021-07-10 20:11:53
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /Cyber-Security-ML-Toolbox/csmt/classifiers/classic/lightgbm.py
'''
from csmt.classifiers.abstract_model import AbstractModel

class LightGBM(AbstractModel):
    def __init__(self,input_size,output_size):
        import lightgbm
        from csmt.estimators.classification.ensemble_tree import EnsembleTree
        model=lightgbm.LGBMClassifier()
        self.classifier=EnsembleTree(model=model,nb_features=input_size, nb_classes=output_size)