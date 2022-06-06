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
from csmt.classifiers.scores import get_class_scores
from csmt.zoopt.bayes_opt.bayesian_optimization import BayesianOptimization


class BayesEnsemble(ClassifierMixin, BaseEstimator):

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
        self.weight=1.0/len(self.model)*np.ones(len(self.model),dtype=float)
    
    def get_distribute(self,max_x,len_distribute):
        x_all=0
        for i in range(len_distribute):
            x_all=x_all+max_x[i]
        distribute=[]
        for i in range(len_distribute):
            distribute.append(max_x[i]/x_all)
        return distribute

    def fit(self, X: np.ndarray, y: np.ndarray,X_val,y_val):
        def get_result(w):
            y_pred_all=np.zeros((X_val.shape[0],self._nb_classes))
            w_new=self.get_distribute(w,len(self.model))
            for i in range(0,len(self.model)):
                y_pred = self.model[i].predict(X_val)
                y_pred_all=y_pred_all+y_pred*w_new[i]
            y_pred_all=np.argmax(y_pred_all, axis=1)
            result=get_class_scores(y_val, y_pred_all)
            #增加惩罚项
            lamda=0
            # print(np.square((np.sum(w)-1)))
            goal=result[0]-lamda*np.square((np.sum(w)-1))
            return goal

        for i in range(len(self.model)):
            self.model[i].train(X, y,X_val,y_val)

        bound=[]
        keys=[]
        for i in range(len(self.model)):
            bound.append([0.01,0.99])
            keys.append('x'+str(i))

        bo = BayesianOptimization(f=get_result,pbounds={'x':bound},random_state=7)
        
        bo.maximize(init_points=10,n_iter=40,distribute=None)
        print(bo.max['params'])
        max_x=np.array([bo.max['params'][key] for key in keys])
        weight_distribute=self.get_distribute(max_x,len(self.model))
        print(weight_distribute)
        self.weight=weight_distribute
        

    def predict(self, x: np.ndarray, **kwargs) -> np.ndarray:
        predictions = self.bayes_ensemble(x)
        return predictions

    def bayes_ensemble(self,X_test):
        y_pred_all=np.zeros((X_test.shape[0],self._nb_classes))
        for i in range(0,len(self.model)):
            y_pred = self.model[i].predict(X_test)
            y_pred_all=y_pred_all+y_pred*self.weight[i]
        return y_pred_all

    @property
    def input_shape(self) -> Tuple[int, ...]:
        """
        Return the shape of one input sample.

        :return: Shape of one input sample.
        """
        return self._input_shape  # type: ignore


