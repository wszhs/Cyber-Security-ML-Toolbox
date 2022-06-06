'''
Author: your name
Date: 2021-06-12 20:01:46
LastEditTime: 2021-07-27 15:10:39
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /Cyber-Security-ML-Toolbox/csmt/estimators/classification/ensemble.py
'''

import numpy as np
from csmt import config
from typing import List, Optional, Union, Tuple, TYPE_CHECKING
from csmt.estimators.classification.classifier import ClassifierMixin
from csmt.estimators.estimator import BaseEstimator


class OriginEnsemble(ClassifierMixin, BaseEstimator):

    def __init__(
        self,
        model: None,
        clip_values: Optional["CLIP_VALUES_TYPE"] = None,
        nb_features: Optional[int] = None,
        nb_classes: Optional[int] = None,
    ) -> None:

        super().__init__(
            model=model,
            clip_values=clip_values
        )
        self._input_shape = (nb_features,)
        self._nb_classes = nb_classes
    

    def fit(self, X: np.ndarray, y: np.ndarray,X_val,y_val):
        for i in range(len(self.model)):
            self.model[i].train(X, y,X_val,y_val)

    def predict(self, x: np.ndarray, **kwargs) -> np.ndarray:

        predictions = self.origin_ensemble(x)
        return predictions

    def origin_ensemble(self,X_test):
        y_pred_arr=np.zeros((len(self.model),X_test.shape[0],self._nb_classes))
        for i in range(0,len(self.model)):
            y_pred_arr[i] = self.model[i].predict(X_test)
        return y_pred_arr

    @property
    def input_shape(self) -> Tuple[int, ...]:
        """
        Return the shape of one input sample.

        :return: Shape of one input sample.
        """
        return self._input_shape  # type: ignore


