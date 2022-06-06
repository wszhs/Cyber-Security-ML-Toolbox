'''
Author: your name
Date: 2021-04-01 16:16:26
LastEditTime: 2021-04-01 21:21:58
LastEditors: your name
Description: In User Settings Edit
FilePath: /Cyber-Security-ML-Toolbox/csmt/classifiers/baseline.py
'''

from csmt.classifiers.abstract_model import AbstractModel
import numpy as np
from scipy import stats

class Baseline(AbstractModel):

    def __init__(self):
        self.classifier = ModeClassifier()

class ModeClassifier:

    def fit(self, X, y):
        self.prediction, _ = stats.mode(y, axis=None)

    def predict(self, X):
        n_observations = len(X)
        predictions = np.array([self.prediction] * n_observations)
        return predictions