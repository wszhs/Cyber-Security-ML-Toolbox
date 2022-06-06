'''
Author: your name
Date: 2021-06-09 14:16:39
LastEditTime: 2021-07-10 19:19:12
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /Cyber-Security-ML-Toolbox/tests/evasion_attack/test_gradient_attack_self.py
'''
'''
Author: your name
Date: 2021-05-27 20:21:24
LastEditTime: 2021-06-07 15:54:41
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /Cyber-Security-ML-Toolbox/tests/test_art/test_rf_attack.py
'''

import sys
ROOT_PATH='/Users/zhanghangsheng/Documents/my_code/Cyber-Security-ML-Toolbox'
sys.path.append(ROOT_PATH)
import numpy as np
from csmt.get_model_data import get_datasets, models_train,parse_arguments,models_train,print_results,models_predict,attack_models_train

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from deepforest import CascadeForestClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import train_test_split

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt

from csmt.estimators.classification.scikitlearn import SklearnClassifier
from csmt.attacks.evasion import FastGradientMethod
from csmt.attacks.evasion import CarliniLInfMethod
from csmt.attacks.evasion import DeepFool,ProjectedGradientDescent,SaliencyMapMethod
from csmt.datasets import load_contagiopdf
from csmt.datasets import load_mnist_flat_zhs,load_datacon

from sklearn import preprocessing
from csmt.classifiers.scores import get_class_scores

def get_model(X_train,y_train):
    # model=SVC()
    model= linear_model.LogisticRegression(max_iter=1000)
    art_classifier = SklearnClassifier(model=model)
    art_classifier.fit(X_train,y_train)
    return art_classifier

def get_adversarial_examples(model,X_test):


    art_classifier = model
    # attack=DeepFool(classifier=art_classifier)
    attack=ProjectedGradientDescent(estimator=art_classifier,eps=0.1,eps_step=0.001)
    # attack = CarliniLInfMethod(classifier=art_classifier,eps=0.1)
    # attack= FastGradientMethod(estimator=art_classifier,eps=0.1, eps_step=0.001,minimal=True)
    # attack=SaliencyMapMethod(art_classifier)
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
    y_pred=np.argmax(model.predict(X_test), axis=1)
    result=get_class_scores(y_test, y_pred)
    print('accuracy f1 precision recall roc_auc')
    print(result)

    X_adv = get_adversarial_examples(model,X_test[0:500])
    y_pred=np.argmax(model.predict(X_adv), axis=1)
    result=get_class_scores(y_test[0:500], y_pred)
    print('accuracy f1 precision recall roc_auc')
    print(result)
