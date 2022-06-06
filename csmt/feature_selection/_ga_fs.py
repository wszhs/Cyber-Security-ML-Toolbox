'''
Author: your name
Date: 2021-04-01 19:28:52
LastEditTime: 2021-05-08 09:44:22
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /Cyber-Security-ML-Toolbox/csmt/feature_selection/_ga_fs.py
'''
import sys
import pandas as pd
import numpy as np
import random
from sklearn import tree
from csmt.classifiers.classic.logistic_regression import LogisticRegression
from csmt.classifiers.classic.decision_tree import DecisionTree
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from csmt.zoopt.GA import GA

def selectGa(X,y):
    def score(my_array):
        index=np.where(my_array==1)[0]
        X_=X[:,index]
        y_=y
        model=LogisticRegression()
        fitness = cross_val_score(model.classifier, X_, y_, cv=2).mean()  # 2次交叉验证
        return -fitness
    chrom_length=X.shape[1]
    ga = GA(func=score, n_dim=chrom_length, size_pop=10, max_iter=10,lb=0,ub=1,precision=1)

    best_x, best_y = ga.run()
    # print('best_x:', best_x, '\n', 'best_y:', best_y)
    return best_x

    