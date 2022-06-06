'''
Author: your name
Date: 2021-05-27 20:21:24
LastEditTime: 2021-06-12 19:10:10
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
from sklearn.svm import SVC, LinearSVC

from sklearn.datasets import load_iris

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt

from art.estimators.classification import SklearnClassifier
from art.estimators.classification import XGBoostClassifier
from art.attacks.evasion import ZooAttack
from art.utils import load_mnist

from csmt.datasets import load_iris_zhs
from csmt.datasets import load_breast_cancer_zhs
from csmt.datasets import load_cicids2017
from csmt.datasets import load_contagiopdf
from csmt.datasets import load_mnist_flat_zhs
from csmt.datasets import load_ctu13
from csmt.datasets import load_datacon
from csmt.datasets import load_drebin
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics

from csmt.classifiers.scores import get_binary_class_scores

def plot_results(model, x_train, y_train, x_train_adv, num_classes):
    fig, axs = plt.subplots(1, num_classes, figsize=(num_classes * 5, 5))

    colors = ['orange', 'blue', 'green']

    for i_class in range(num_classes):

        # Plot difference vectors
        for i in range(y_train[y_train == i_class].shape[0]):
            x_1_0 = x_train[y_train == i_class][i, 0]
            x_1_1 = x_train[y_train == i_class][i, 1]
            x_2_0 = x_train_adv[y_train == i_class][i, 0]
            x_2_1 = x_train_adv[y_train == i_class][i, 1]
            if x_1_0 != x_2_0 or x_1_1 != x_2_1:
                axs[i_class].plot([x_1_0, x_2_0], [x_1_1, x_2_1], c='black', zorder=1)

        # Plot benign samples
        for i_class_2 in range(num_classes):
            axs[i_class].scatter(x_train[y_train == i_class_2][:, 0], x_train[y_train == i_class_2][:, 1], s=20,
                                 zorder=2, c=colors[i_class_2])
        axs[i_class].set_aspect('equal', adjustable='box')

        # Show predicted probability as contour plot
        h = .01
        x_min, x_max = 0, 1
        y_min, y_max = 0, 1

        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

        Z_proba = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])
        Z_proba = Z_proba[:, i_class].reshape(xx.shape)
        im = axs[i_class].contourf(xx, yy, Z_proba, levels=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                                   vmin=0, vmax=1)
        if i_class == num_classes - 1:
            cax = fig.add_axes([0.95, 0.2, 0.025, 0.6])
            plt.colorbar(im, ax=axs[i_class], cax=cax)

        # Plot adversarial samples
        for i in range(y_train[y_train == i_class].shape[0]):
            x_1_0 = x_train[y_train == i_class][i, 0]
            x_1_1 = x_train[y_train == i_class][i, 1]
            x_2_0 = x_train_adv[y_train == i_class][i, 0]
            x_2_1 = x_train_adv[y_train == i_class][i, 1]
            if x_1_0 != x_2_0 or x_1_1 != x_2_1:
                axs[i_class].scatter(x_2_0, x_2_1, zorder=2, c='red', marker='X')
        axs[i_class].set_xlim((x_min, x_max))
        axs[i_class].set_ylim((y_min, y_max))

        axs[i_class].set_title('class ' + str(i_class))
        axs[i_class].set_xlabel('feature 1')
        axs[i_class].set_ylabel('feature 2')
    plt.show()


def get_model(X_train,y_train):
    # Create and fit RandomForestClassifier
    # model = RandomForestClassifier()
    # model= linear_model.LogisticRegression(max_iter=1000)
    # model=KNeighborsClassifier(n_neighbors=5,weights='uniform',algorithm='auto')
    # model=tree.DecisionTreeClassifier()
    model=xgboost.XGBClassifier()
    # model=LinearSVC()
    # model=SVC(probability=True)
    # art_classifier = SklearnClassifier(model=model)
    model.fit(X_train,y_train)
    return model

def get_adversarial_examples(model,X_test):

    # Create ART classfier for scikit-learn RandomForestClassifier
    
    art_classifier = XGBoostClassifier(model=model, nb_features=X_test.shape[1], nb_classes=2)

    zoo = ZooAttack(classifier=art_classifier, confidence=0.0, targeted=False, learning_rate=1e-1, max_iter=30,
                binary_search_steps=20, initial_const=1e-3, abort_early=True, use_resize=False, 
                use_importance=False, nb_parallel=20, batch_size=1, variable_h=0.1)
    X_adv = zoo.generate(X_test)

    for i in range(10):
        dis=np.linalg.norm(X_adv[i]-X_test[i],ord=2)
        print(dis)
    return X_adv

X,y=load_contagiopdf()
scaler1 = preprocessing.MinMaxScaler().fit(X)
X = scaler1.transform(X)
X_train,X_test,y_train,y_test = train_test_split(X,y, stratify=y,train_size=0.8, test_size=0.2,random_state=42)
y_train,y_test=y_train.values,y_test.values


model=get_model(X_train,y_train)
y_pred=model.predict(X_test)
result=get_binary_class_scores(y_test, y_pred)
print('accuracy f1 precision recall roc_auc')
print(result)

X_adv = get_adversarial_examples(model,X_test[0:100])
y_pred=model.predict(X_adv)
result=get_binary_class_scores(y_test[0:100], y_pred)
print('accuracy f1 precision recall roc_auc')
print(result)
