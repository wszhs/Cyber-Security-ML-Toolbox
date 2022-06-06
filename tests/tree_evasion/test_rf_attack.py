'''
Author: your name
Date: 2021-04-19 09:51:59
LastEditTime: 2021-05-27 19:08:06
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /Cyber-Security-ML-Toolbox/tests/test_rf_attack.py
'''
import sys
sys.path.append("/Users/zhanghangsheng/Documents/my_code/Cyber-Security-ML-Toolbox")

import re
import IPython, graphviz
import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import pandas as pd
from csmt.datasets import load_iris_zhs
from csmt.datasets import load_cicandmal2017
from csmt.datasets import load_contagiopdf
from csmt.datasets import load_mnist_flat_zhs

from sklearn.ensemble import RandomForestClassifier
from copy import deepcopy
from tqdm import tqdm
import matplotlib.pyplot as plt

from tree_evasion.core import *
from tree_evasion.tree import *

SEED = 41
np.random.seed(SEED)

#export
_all_ = ['SymbolicInstance', 'SymbolicPredictionSolver', 'CoordinateDescent']

#export
class SymbolicInstance:
    def __init__(self, constraints):
        self.constraints = constraints  
        self.x_prime_constraints = {}
        self.k = None
    
    def diff_instances(self):
        ck  = list(set(self.x_prime_constraints.keys()) & set(self.constraints.keys()))
        res = 0
        
        for k in ck:
            if (k in self.x_prime_constraints) and (k in self.constraints):
                if self.constraints[k] != self.x_prime_constraints[k]:
                    res += 1
        return res
    
    def is_feasible(self, predicate, res):
        feat, threshold = predicate
        
        d = self.diff_instances()
        
        if d == 1:
            if feat not in self.constraints: return False
            if (threshold > self.constraints[feat][0]) and (threshold <= self.constraints[feat][1]): return True
            return False
        else:
            if feat not in self.constraints: return False
        
        return True
    
    def update(self, predicate, res):
        feat, threshold = predicate
        self.k = feat
        
        if res:
            self.x_prime_constraints[feat] = [-np.inf, threshold]
        else:
            self.x_prime_constraints[feat] = [threshold, np.inf]
    
    def is_changed(self): 
        return self.diff_instances() == 1
    
    def get_perturbation(self): return (self.k, self.x_prime_constraints[self.k])

#export
class SymbolicPredictionSolver:
    def solve(self, tree, index, s):
        self.l = []
        self.symbolic_prediction(tree, index, s)
        return self.l

    def symbolic_prediction(self, tree, index, s):
        if tree.is_leaf(index):
            if s.is_changed():        
                k, half_open_interval = s.get_perturbation()
                self.l.extend([[k, half_open_interval, tree.prediction(index)]])
        else:
            if s.is_feasible(tree.predicate(index), True):
                s_t = deepcopy(s)
                s_t.update(tree.predicate(index), True)
                self.symbolic_prediction(tree, tree.get_left_child_index(index), s_t)
            if s.is_feasible(tree.predicate(index), False):
                s.update(tree.predicate(index), False)
                self.symbolic_prediction(tree, tree.get_right_child_index(index), s)

#export
class CoordinateDescent:
    @classmethod
    def get_pairs(cls, clf, Xte):
        pairs = []

        for i in tqdm(range(100)):
            forest_prediction_intervals = []
            x = Xte[i:i+1]

            # if this instance is correctly predicted by the classifier or not
            p = clf.predict(x)[0]

            if p != y_test[i]:
                continue

            for tidx in range(len(clf.estimators_)):
                t = Tree(clf.estimators_[tidx].tree_) 

                x_constraints = t.constraints(x)
                s = SymbolicInstance(x_constraints)
                solver = SymbolicPredictionSolver()
                perturbations = solver.solve(t, 0, s)

                # index of the leaf node for x
                x_path = clf.estimators_[tidx].tree_.decision_path(x.astype(np.float32)).toarray().ravel()
                leaf_node_index = np.where(x_path == 1)[0][-1]
                pred = t.prediction(leaf_node_index)

                prediction_intervals = []
                for index, p in enumerate(perturbations):
                    prediction_intervals.append([p[0], p[1], p[2] - pred])

                forest_prediction_intervals.extend(prediction_intervals)

            max_interval = sorted(forest_prediction_intervals, key=lambda x: x[2], reverse=True)[0]
            class_ = 0 if clf.predict(x)[0] == '2' else 1

            x_prime = Xte[i].copy()
            x_prime[max_interval[0]] = np.random.uniform(0 if max_interval[1][0] == -np.inf else max_interval[1][0], 
                                                         1 if max_interval[1][1] == np.inf else max_interval[1][1])

            perturbed_class = 0 if clf.predict(x_prime.reshape(1, -1))[0] == '2' else 1

            if class_ != perturbed_class:
                print(f'class: {class_}, perturbed_class: {perturbed_class}')
                pairs.append((x, x_prime))

        return pairs

# X_train,y_train,X_test,y_test=load_contagiopdf('all_binary')
# X_train,y_train,X_test,y_test=load_iris_zhs()
# X_train, X_, X_test, y_train, y_, y_test = get_mnist_dataset(SEED)

X_train,y_train,X_test,y_test=load_mnist_flat_zhs()
# X_train, Xva, X_test, y_train, yva, y_test = get_mnist_dataset(SEED)

clf = RandomForestClassifier(n_estimators=5, max_depth=4, random_state=SEED, n_jobs=-1)
clf.fit(X_train, y_train)

print(clf.score(X_test, y_test))

pairs = CoordinateDescent.get_pairs(clf, X_test)
print(len(pairs))