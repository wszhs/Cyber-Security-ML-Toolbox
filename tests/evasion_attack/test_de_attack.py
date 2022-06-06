'''
Author: your name
Date: 2021-06-07 15:43:26
LastEditTime: 2021-06-10 10:06:34
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /Cyber-Security-ML-Toolbox/tests/evasion_attack/test_de_attack.py
'''
'''
Author: your name
Date: 2021-05-27 20:21:24
LastEditTime: 2021-06-07 15:41:59
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /Cyber-Security-ML-Toolbox/tests/test_art/test_rf_attack.py
'''
import sys
sys.path.append("/Users/zhanghangsheng/Documents/my_code/Cyber-Security-ML-Toolbox")

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import linear_model
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from deepforest import CascadeForestClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import train_test_split
import xgboost

from sklearn.datasets import load_iris

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt

from art.estimators.classification import SklearnClassifier
from art.estimators.classification import XGBoostClassifier
from art.attacks.evasion import ZooAttack
from art.attacks.evasion import FastGradientMethod,CarliniLInfMethod,ProjectedGradientDescent,DeepFool,CarliniL2Method
from csmt.attacks.evasion.gradient import GradientEvasionAttack
from csmt.attacks.evasion.de import DEEvasionAttack
from art.utils import load_mnist

from csmt.classifiers.scores import get_class_scores

import sys
ROOT_PATH='/Users/zhanghangsheng/Documents/my_code/Cyber-Security-ML-Toolbox'
sys.path.append(ROOT_PATH)
import numpy as np
from csmt.get_model_data import get_datasets, models_train,parse_arguments,models_train,print_results,models_predict,attack_models_train
from sklearn import metrics

from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler

def get_model(X_train,y_train):
    model= linear_model.LogisticRegression(max_iter=1000)
    model.fit(X_train,y_train)
    return model

def get_adversarial_examples(model,X_test):
    def get_score(p):
        score=model.predict_proba(p.reshape(1,-1))
        return -score[0][0]

    attack=DEEvasionAttack(get_score)
    X_adv = attack.generate(X_test)

    for i in range(10):
        dis=np.linalg.norm(X_adv[i]-X_test[i],ord=2)
        print(dis)

    return X_adv

if __name__=='__main__':
    arguments = sys.argv[1:]
    options = parse_arguments(arguments)
    datasets_name=options.datasets
    orig_models_name=options.algorithms
    evasion_algorithm=options.evasion_algorithm
    X_train,y_train,X_val,y_val,X_test,y_test,mask=get_datasets(options)
    X_train,y_train,X_test,y_test=X_train,y_train,X_test,y_test

    model=get_model(X_train,y_train)
    y_pred=model.predict(X_test)
    result=get_class_scores(y_test, y_pred)
    print('accuracy f1 precision recall roc_auc')
    print(result)

    X_adv = get_adversarial_examples(model,X_test[0:100])
    y_pred=model.predict(X_adv)
    result=get_class_scores(y_test[0:100], y_pred)
    print('accuracy f1 precision recall roc_auc')
    print(result)