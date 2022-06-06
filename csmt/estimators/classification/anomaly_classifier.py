'''
Author: your name
Date: 2021-07-15 18:01:42
LastEditTime: 2021-07-20 19:47:36
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /Cyber-Security-ML-Toolbox/csmt/estimators/classification/isolationForest.py
'''

from itertools import combinations
import numpy as np
from typing import List, Optional, Union, Tuple, TYPE_CHECKING
from csmt.estimators.classification.classifier import ClassifierMixin
from csmt.estimators.estimator import BaseEstimator
from sklearn.preprocessing import MinMaxScaler
import matplotlib
from numpy import percentile

matplotlib.use('TKAgg')
import matplotlib.pyplot as plt


class AnomalyClassifeir(ClassifierMixin, BaseEstimator):

    def __init__(
        self,
        model: None,
        clip_values: Optional["CLIP_VALUES_TYPE"] = None,
        nb_features: Optional[int] = None,
        nb_classes: Optional[int] = None,
        contamination : Optional[float] = None
    ) -> None:

        super().__init__(
            model=model,
            clip_values=clip_values
        )
        self._input_shape = (nb_features,)
        self._nb_classes = nb_classes
        self.threshold=0
        self.decision_scores_=None
        self.contamination=contamination
    

    def fit(self, x: np.ndarray, y: np.ndarray,X_val,y_val):
        X_train=x[y==0]
        y_train=y[y==0]
        self.model.fit(X_train)

        self.decision_scores_=self.model.decision_function(X_train)
        self.threshold = percentile(self.decision_scores_, 100 * (1 - self.contamination))


    def predict(self, x: np.ndarray, **kwargs) -> np.ndarray:
        anomaly_scores=self.model.decision_function(x)
        anomaly_scores= anomaly_scores.reshape(-1,1)
        y_pred=np.hstack((-(anomaly_scores-self.threshold),(anomaly_scores-self.threshold)))
        return y_pred
        
    def decision_function(self,x):
        return self.model.decision_function(x)


    @property
    def input_shape(self) -> Tuple[int, ...]:
        """
        Return the shape of one input sample.

        :return: Shape of one input sample.
        """
        return self._input_shape  # type: ignore



