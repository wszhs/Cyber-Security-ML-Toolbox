'''
Author: your name
Date: 2021-04-02 11:16:29
LastEditTime: 2021-04-02 11:22:15
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /Cyber-Security-ML-Toolbox/tests/test_baggingClassifier.py
'''
from csmt.datasets import load_breast_cancer_zhs
from csmt.datasets import load_iris_zhs
from csmt.datasets import load_contagiopdf
from csmt.datasets import load_cicandmal2017
from sklearn.model_selection import cross_val_score   # 10折交叉验证评价模型
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline   # 管道简化工作流
from sklearn.preprocessing import StandardScaler  # 标准化数据

import numpy as np

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


X_train,y_train,X_test,y_test=load_contagiopdf('all_binary')
## 我们使用单一决策树分类：
tree = DecisionTreeClassifier(criterion='entropy',random_state=1,max_depth=None)   #选择决策树为基本分类器
from sklearn.metrics import accuracy_score
tree = tree.fit(X_train,y_train)
y_train_pred = tree.predict(X_train)
y_test_pred = tree.predict(X_test)
tree_train = accuracy_score(y_train,y_train_pred)
tree_test = accuracy_score(y_test,y_test_pred)
print('Decision tree train/test accuracies %.3f/%.3f' % (tree_train,tree_test))

## 我们使用BaggingClassifier分类：
from sklearn.ensemble import BaggingClassifier
tree = DecisionTreeClassifier(criterion='entropy',random_state=1,max_depth=None)   #选择决策树为基本分类器
bag = BaggingClassifier(base_estimator=tree,n_estimators=500,max_samples=1.0,max_features=1.0,bootstrap=True,
                        bootstrap_features=False,n_jobs=1,random_state=1)
from sklearn.metrics import accuracy_score
bag = bag.fit(X_train,y_train)
y_train_pred = bag.predict(X_train)
y_test_pred = bag.predict(X_test)
bag_train = accuracy_score(y_train,y_train_pred)
bag_test = accuracy_score(y_test,y_test_pred)
print('Bagging train/test accuracies %.3f/%.3f' % (bag_train,bag_test))

## 我们来对比下这两个分类方法上的差异
# x_min = X_train[:, 0].min() - 1
# x_max = X_train[:, 0].max() + 1
# y_min = X_train[:, 1].min() - 1
# y_max = X_train[:, 1].max() + 1
# xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),np.arange(y_min, y_max, 0.1))
# f, axarr = plt.subplots(nrows=1, ncols=2,sharex='col',sharey='row',figsize=(12, 6))
# for idx, clf, tt in zip([0, 1],[tree, bag],['Decision tree', 'Bagging']):
#     clf.fit(X_train, y_train)
#     Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
#     Z = Z.reshape(xx.shape)
#     axarr[idx].contourf(xx, yy, Z, alpha=0.3)
#     axarr[idx].scatter(X_train[y_train==0, 0],X_train[y_train==0, 1],c='blue', marker='^')
#     axarr[idx].scatter(X_train[y_train==1, 0],X_train[y_train==1, 1],c='green', marker='o')
#     axarr[idx].set_title(tt)
# axarr[0].set_ylabel('Alcohol', fontsize=12)
# plt.tight_layout()
# plt.text(0, -0.2,s='OD280/OD315 of diluted wines',ha='center',va='center',fontsize=12,transform=axarr[1].transAxes)
# plt.show()