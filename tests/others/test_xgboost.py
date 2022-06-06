'''
Author: your name
Date: 2021-04-02 14:19:08
LastEditTime: 2021-04-19 09:28:10
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /Cyber-Security-ML-Toolbox/tests/test_xgboost.py
'''
import numpy as np
import pandas as pd 
import xgboost as xgb
import sys
sys.path.append("/Users/zhanghangsheng/Documents/my_code/Cyber-Security-ML-Toolbox")
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from csmt.datasets import load_breast_cancer_zhs
from csmt.datasets import load_iris_zhs

from sklearn.datasets import load_iris
import xgboost as xgb
from xgboost import plot_importance
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score   # 准确率
# 加载样本数据集
X_train,y_train,X_test,y_test=load_iris_zhs()
# 算法参数
params = {
    'booster': 'gbtree',
    'objective': 'multi:softmax',
    'num_class': 3,
    'gamma': 0.1,
    'max_depth': 6,
    'lambda': 2,
    'subsample': 0.7,
    'colsample_bytree': 0.75,
    'min_child_weight': 3,
    'silent': 0,
    'eta': 0.1,
    'seed': 1,
    'nthread': 4,
}

plst = list(params.items())

dtrain = xgb.DMatrix(X_train, y_train) # 生成数据集格式
num_rounds = 500
model = xgb.train(plst, dtrain, num_rounds) # xgboost模型训练

# 对测试集进行预测
dtest = xgb.DMatrix(X_test)
y_pred = model.predict(dtest)

# 计算准确率
accuracy = accuracy_score(y_test,y_pred)
print("accuarcy: %.2f%%" % (accuracy*100.0))

# # 显示重要特征
# plot_importance(model)
# plt.show()

# #调参
# import xgboost as xgb
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.model_selection import GridSearchCV
# from sklearn.metrics import roc_auc_score

# parameters = {
#               'max_depth': [5, 10, 15, 20, 25],
#               'learning_rate': [0.01, 0.02, 0.05, 0.1, 0.15],
#               'n_estimators': [500, 1000, 2000, 3000, 5000],
#               'min_child_weight': [0, 2, 5, 10, 20],
#               'max_delta_step': [0, 0.2, 0.6, 1, 2],
#               'subsample': [0.6, 0.7, 0.8, 0.85, 0.95],
#               'colsample_bytree': [0.5, 0.6, 0.7, 0.8, 0.9],
#               'reg_alpha': [0, 0.25, 0.5, 0.75, 1],
#               'reg_lambda': [0.2, 0.4, 0.6, 0.8, 1],
#               'scale_pos_weight': [0.2, 0.4, 0.6, 0.8, 1]

# }

# xlf = xgb.XGBClassifier(max_depth=10,
#             learning_rate=0.01,
#             n_estimators=2000,
#             silent=True,
#             objective='multi:softmax',
#             num_class=3 ,          
#             nthread=-1,
#             gamma=0,
#             min_child_weight=1,
#             max_delta_step=0,
#             subsample=0.85,
#             colsample_bytree=0.7,
#             colsample_bylevel=1,
#             reg_alpha=0,
#             reg_lambda=1,
#             scale_pos_weight=1,
#             seed=0,
#             missing=None)

# gs = GridSearchCV(xlf, param_grid=parameters, scoring='accuracy', cv=3)
# gs.fit(X_train, y_train)

# print("Best score: %0.3f" % gs.best_score_)
# print("Best parameters set: %s" % gs.best_params_ )
