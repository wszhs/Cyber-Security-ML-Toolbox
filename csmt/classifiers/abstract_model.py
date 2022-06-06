'''
Author: your name
Date: 2021-03-24 16:09:20
LastEditTime: 2021-07-10 19:53:33
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /NashHE/ml_ids/abstract_model.py
'''
from joblib import dump, load
import numpy as np

class AbstractModel:
    """
    Base model that all other models should inherit from.
    Expects that classifier algorithm is initialized during construction.
    """

    def train(self, X_train, y_train,X_val,y_val):
        self.classifier.fit(X_train, y_train,X_val,y_val)

    def predict(self, X):
        return self.classifier.predict(X)

    def predict_label(self, X):
        pred=self.classifier.predict(X)
        label=np.argmax(pred, axis=1)
        return label

    #仅限异常检测模型
    def predict_anomaly(self,X,y):
        # from sklearn.metrics import roc_auc_score,precision_score
        from csmt.classifiers.anomaly_detection.pyod.utils.data import evaluate_print
        anomaly_scores=self.classifier.decision_function(X)
        roc=evaluate_print('model', y, anomaly_scores)
        # roc=np.round(roc_auc_score(y, anomaly_scores), decimals=4)
        print(roc)

    def save(self, path):
        dump(self.classifier, path)

    def load(self, path):
        self.classifier = load(path)
