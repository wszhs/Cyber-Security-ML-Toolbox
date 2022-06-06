'''
Author: your name
Date: 2021-06-12 20:01:46
LastEditTime: 2021-07-27 14:57:09
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /Cyber-Security-ML-Toolbox/csmt/estimators/classification/ensemble.py
'''

import numpy as np
from csmt import config
from typing import List, Optional, Union, Tuple, TYPE_CHECKING
from csmt.estimators.classification.classifier import ClassifierMixin
from csmt.estimators.estimator import BaseEstimator


class StackingEnsemble(ClassifierMixin, BaseEstimator):

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

        predictions = self.stacking_ensemble(x)
        return predictions

    def stacking_ensemble(self,X_test):
        weight=1.0/len(self.model)*np.ones(len(self.model),dtype=float)
        y_pred_all=np.zeros((X_test.shape[0],self._nb_classes))
        for i in range(0,len(self.model)):
            y_pred = self.model[i].predict(X_test)
            y_pred_all=y_pred_all+y_pred*weight[i]
        return y_pred_all

    @property
    def input_shape(self) -> Tuple[int, ...]:
        """
        Return the shape of one input sample.

        :return: Shape of one input sample.
        """
        return self._input_shape  # type: ignore


