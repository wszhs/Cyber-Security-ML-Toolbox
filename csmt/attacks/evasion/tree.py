'''
Author: your name
Date: 2021-04-07 09:20:13
LastEditTime: 2021-05-08 18:13:45
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /Cyber-Security-ML-Toolbox/csmt/attacks/evasion/tree.py
'''
from art.estimators.classification import SklearnClassifier
from art.attacks.evasion.decision_tree_attack import DecisionTreeAttack
from art.classifiers import SklearnClassifier
from csmt.attacks.evasion.abstract_evasion import AbstractEvasion
import numpy as np

class TreeEvasionAttack(AbstractEvasion):
    def __init__(self,estimator):
        self.model=estimator
        
    def generate(self,X,y):
        X_adv_path=np.zeros((X.shape[0],2,X.shape[1]))
        art_classifier = SklearnClassifier(model=self.model,clip_values=(np.zeros(X.shape[1]), np.ones(X.shape[1])))
        attack= DecisionTreeAttack(art_classifier)
        X_adv = attack.generate(X)
        y_adv=y
        return X_adv,y_adv,X_adv_path