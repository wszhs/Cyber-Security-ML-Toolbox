'''
Author: your name
Date: 2021-05-27 20:21:24
LastEditTime: 2021-07-12 11:15:18
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /Cyber-Security-ML-Toolbox/tests/test_art/test_rf_attack.py
'''
import sys
sys.path.append("/Users/zhanghangsheng/Documents/my_code/Cyber-Security-ML-Toolbox")
from sklearn.ensemble import RandomForestClassifier
from sklearn import linear_model
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from deepforest import CascadeForestClassifier
from sklearn.model_selection import train_test_split
import xgboost
import lightgbm
import catboost
from sklearn.svm import SVC, LinearSVC

from sklearn.datasets import load_iris

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt

from csmt.estimators.classification.scikitlearn import SklearnClassifier
from csmt.estimators.classification.xgboost import XGBoostClassifier
from csmt.estimators.classification.lightgbm import LightGBMClassifier
from csmt.estimators.classification.catboost import CatBoostCSMTClassifier
from csmt.estimators.classification.ensemble_tree import EnsembleTree
from csmt.estimators.classification.ensemble import Ensemble
from csmt.attacks.evasion.zoo import ZooAttack
from csmt.utils import load_mnist

from csmt.datasets import load_iris_zhs
from csmt.datasets import load_breast_cancer_zhs
from csmt.datasets import load_cicandmal2017
from csmt.datasets import load_contagiopdf
from csmt.datasets import load_mnist_flat_zhs
from csmt.datasets import load_ctu13,load_dohbrw
from csmt.datasets import load_datacon
from csmt.datasets import load_drebin
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics

from csmt.classifiers.scores import get_class_scores


def get_model(X_train,y_train):
    # Create and fit RandomForestClassifier
    # model = RandomForestClassifier()
    model= linear_model.LogisticRegression(max_iter=1000)
    # model=tree.DecisionTreeClassifier(random_state=0)
    # model=KNeighborsClassifier(n_neighbors=5)
    # model=LinearSVC()
    # model=SVC(probability=True)
    # model=catboost.CatBoostClassifier(logging_level='Silent')
    # model=lightgbm.LGBMClassifier()
    # model=xgboost.XGBClassifier()
    # art_classifier=EnsembleTree(model=model,nb_features=X_train.shape[1], nb_classes=2)
    # model=CascadeForestClassifier(random_state=1,verbose=0)
    # art_classifier=EnsembleTree(model=model,nb_features=X_train.shape[1])
    art_classifier = SklearnClassifier(model=model)
    # models=[linear_model.LogisticRegression(max_iter=1000),tree.DecisionTreeClassifier(random_state=0)]
    # art_classifier=Ensemble(model=models)

    art_classifier.fit(X_train,y_train)
    return art_classifier


def get_model2(X_train,y_train):
    # Create and fit RandomForestClassifier
    # model = RandomForestClassifier()
    model= linear_model.LogisticRegression(max_iter=1000)
    # model=tree.DecisionTreeClassifier(random_state=0)
    # model=KNeighborsClassifier(n_neighbors=5)
    # model=LinearSVC()
    # model=SVC(probability=True)
    # model=catboost.CatBoostClassifier(logging_level='Silent')
    # model=lightgbm.LGBMClassifier()
    # model=xgboost.XGBClassifier()
    # model=CascadeForestClassifier(random_state=1,verbose=0)
    # art_classifier=EnsembleTree(model=model,nb_features=X_test.shape[1], nb_classes=2)
    art_classifier = SklearnClassifier(model=model)

    art_classifier.fit(X_train,y_train)
    return art_classifier



def get_adversarial_examples(model,X_test):

    # Create ART classfier for scikit-learn RandomForestClassifier
    art_classifier = model
    zoo = ZooAttack(classifier=art_classifier, confidence=0.0, targeted=False, learning_rate=1e-1, max_iter=30,
                binary_search_steps=10, initial_const=1e-3, abort_early=True, use_resize=False, 
                use_importance=False, nb_parallel=20, batch_size=1, variable_h=0.1)
    X_adv = zoo.generate(X_test)

    for i in range(20):
        dis=np.linalg.norm(X_adv[i]-X_test[i],ord=2)
        print(dis)
    return X_adv

X,y,mask=load_contagiopdf()
scaler1 = preprocessing.MinMaxScaler().fit(X)
X = scaler1.transform(X)
X_train,X_test,y_train,y_test = train_test_split(X,y, stratify=y,train_size=0.8, test_size=0.2,random_state=42)
y_train,y_test=y_train.values,y_test.values

# X,y,mask=load_datacon("all_binary")
# scaler1 = preprocessing.MinMaxScaler().fit(X)
# X = scaler1.transform(X)
# X_train,X_test,y_train,y_test = train_test_split(X,y, stratify=y,train_size=0.8, test_size=0.2,random_state=42)
# y_train,y_test=y_train.values,y_test.values

# X_train,y_train,X_test,y_test=load_mnist_flat_zhs()
# X_train,y_train,X_test,y_test=X_train[0:2000],y_train[0:2000],X_test[0:500],y_test[0:500]

model=get_model(X_train,y_train)
model2=get_model2(X_train,y_train)

y_pred=np.argmax(model.predict(X_test), axis=1)
result=get_class_scores(y_test, y_pred)
print('accuracy f1 precision recall roc_auc')
print(result)

X_adv = get_adversarial_examples(model,X_test)
y_pred=np.argmax(model2.predict(X_adv), axis=1)
result=get_class_scores(y_test[0:100], y_pred)
print('accuracy f1 precision recall roc_auc')
print(result)
