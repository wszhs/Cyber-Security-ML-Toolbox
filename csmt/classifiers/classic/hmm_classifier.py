'''
Author: your name
Date: 2021-04-19 11:28:27
LastEditTime: 2021-07-10 19:35:02
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /Cyber-Security-ML-Toolbox/csmt/classifiers/classic/hmm_classifier.py
'''
import operator
from copy import copy
from scipy.special import softmax


class HMM_classifier():
    def __init__(self, base_hmm_model):
        self.models = {}
        self.hmm_model = base_hmm_model

    def fit(self, X, Y):
        """
        X: input sequence [[[x1,x2,.., xn]...]]
        Y: output classes [1, 2, 1, ...]
        """
        # print("Detect classes:", set(Y))
        # print("Prepare datasets...")
        X_Y = {}
        X_lens = {}
        for c in set(Y):
            X_Y[c] = []
            X_lens[c] = []

        for x, y in zip(X, Y):
            X_Y[y].extend(x)
            X_lens[y].append(len(x))

        for c in set(Y):
            # print("Fit HMM for", c, " class")
            hmm_model = copy(self.hmm_model)
            hmm_model.fit(X_Y[c], X_lens[c])
            self.models[c] = hmm_model

    def _predict_scores(self, X):

        """
        X: input sample [[x1,x2,.., xn]]
        Y: dict with log likehood per class
        """
        X_seq = []
        X_lens = []
        for x in X:
            X_seq.extend(x)
            X_lens.append(len(x))

        scores = {}
        for k, v in self.models.items():
            scores[k] = v.score(X)
        return scores

    def predict_proba(self, X):
        """
        X: input sample [[x1,x2,.., xn]]
        Y: dict with probabilities per class
        """
        pred = self._predict_scores(X)
        return pred

    def predict(self, X):
        """
        X: input sample [[x1,x2,.., xn]]
        Y: predicted class label
        """
        pred = self.predict_proba(X)

        return max(pred.items(), key=operator.itemgetter(1))[0]
    