'''
Author: your name
Date: 2021-04-02 15:07:05
LastEditTime: 2021-04-02 15:29:06
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /Cyber-Security-ML-Toolbox/tests/test_bayes_opt.py
'''
import sys
sys.path.append("/Users/zhanghangsheng/Documents/my_code/Cyber-Security-ML-Toolbox")
from csmt.datasets import load_breast_cancer_zhs
from csmt.datasets import load_iris_zhs
from csmt.datasets import load_contagiopdf
from csmt.datasets import load_nslkdd
from csmt.datasets import load_cicandmal2017
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from bayes_opt import BayesianOptimization
from sklearn.model_selection import GridSearchCV
import numpy as np

# 产生随机分类数据集，10个特征， 2个类别
X_train, y_train = make_classification(n_samples=1000,n_features=10,n_classes=2)
# X_train,y_train,X_test,y_test=load_iris_zhs()

# 先看看不调参的结果：
rf = RandomForestClassifier()
print(np.mean(cross_val_score(rf, X_train, y_train, cv=20, scoring='roc_auc')))

# bayes调参初探
def rf_cv(n_estimators, min_samples_split, max_features, max_depth):
    val = cross_val_score(
        RandomForestClassifier(n_estimators=int(n_estimators),
            min_samples_split=int(min_samples_split),
            max_features=min(max_features, 0.999), # float
            max_depth=int(max_depth),
            random_state=2
        ),
        X_train, y_train, scoring='roc_auc', cv=5
    ).mean()
    return val

# 实例化一个bayes优化对象
rf_bo = BayesianOptimization(
        rf_cv,
        {'n_estimators': (10, 250),
        'min_samples_split': (2, 25),
        'max_features': (0.1, 0.999),
        'max_depth': (5, 15)}
    )
# 最优化
rf_bo.maximize()

print(rf_bo.max)

cross_val_score(
        RandomForestClassifier(n_estimators=249,
            min_samples_split=24,
            max_features=0.399, # float
            max_depth=13,
            random_state=2
        ),
        X_train, y_train, scoring='roc_auc', cv=5
    ).mean()