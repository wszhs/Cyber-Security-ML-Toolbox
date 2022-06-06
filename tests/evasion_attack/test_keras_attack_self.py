'''
Author: your name
Date: 2021-06-09 17:00:55
LastEditTime: 2021-06-22 11:00:29
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /Cyber-Security-ML-Toolbox/tests/evasion_attack/test_keras_attack.py
'''
'''
Author: your name
Date: 2021-05-27 20:21:24
LastEditTime: 2021-06-07 15:54:41
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /Cyber-Security-ML-Toolbox/tests/test_art/test_rf_attack.py
'''
from logging import log
import sys
sys.path.append("/Users/zhanghangsheng/Documents/my_code/Cyber-Security-ML-Toolbox")

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import linear_model
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
# from deepforest import CascadeForestClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import train_test_split
import xgboost

from sklearn.datasets import load_iris

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt

from csmt.estimators.classification.keras import KerasClassifier
from csmt.attacks.evasion import FastGradientMethod,ProjectedGradientDescent,SaliencyMapMethod

from csmt.classifiers.classic.logistic_regression import LogisticRegression
from csmt.classifiers.keras.multi_layer_perceptron import MultiLayerPerceptronKeras
from csmt.classifiers.keras.lstm import LSTMKeras

from csmt.datasets import load_contagiopdf,load_mnist_flat_zhs,load_datacon,load_cicandmal2017
from sklearn import metrics

from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from csmt.classifiers.scores import get_binary_class_scores

def get_model(X_train,y_train):
    model=MultiLayerPerceptronKeras(input_size=X_train.shape[1])
    # model=LSTMKeras(input_size=X_train.shape[1])
    art_classifier = KerasClassifier(model.model)
    art_classifier.fit(X_train,y_train)
    return art_classifier

def get_adversarial_examples(model,X_test):

    art_classifier = model
    attack= FastGradientMethod(estimator=art_classifier,eps=0.3, eps_step=0.001,minimal=True)
    # attack=ProjectedGradientDescent(estimator=art_classifier,eps=0.05,eps_step=0.001,max_iter=1000)
    # attack=SaliencyMapMethod(art_classifier)
    X_adv = attack.generate(X_test)

    for i in range(10):
        dis=np.linalg.norm(X_adv[i]-X_test[i],ord=2)
        print(dis)

    return X_adv

X,y=load_contagiopdf()
scaler1 = preprocessing.MinMaxScaler().fit(X)
X = scaler1.transform(X)
X_train,X_test,y_train,y_test = train_test_split(X,y, stratify=y,train_size=0.8, test_size=0.2,random_state=42)
y_train,y_test=y_train.values,y_test.values

# X,y,mask=load_cicandmal2017("all_binary")
# scaler1 = preprocessing.MinMaxScaler().fit(X)
# X = scaler1.transform(X)
# X_train,X_test,y_train,y_test = train_test_split(X,y, stratify=y,train_size=0.8, test_size=0.2,random_state=42)
# y_train,y_test=y_train.values,y_test.values

# X_train,y_train,X_test,y_test=load_mnist_flat_zhs()
# X_train,y_train,X_test,y_test=X_train[0:2000],y_train[0:2000],X_test[0:500],y_test[0:500]

model=get_model(X_train,y_train)
y_pred=np.argmax(model.predict(X_test), axis=1)
result=get_binary_class_scores(y_test, y_pred)
print('accuracy f1 precision recall roc_auc')
print(result)

X_adv = get_adversarial_examples(model,X_test[0:100])
y_pred=np.argmax(model.predict(X_adv), axis=1)
result=get_binary_class_scores(y_test[0:100], y_pred)
print('accuracy f1 precision recall roc_auc')
print(result)
