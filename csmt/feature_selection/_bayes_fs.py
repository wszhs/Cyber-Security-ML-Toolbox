'''
Author: your name
Date: 2021-04-25 15:42:02
LastEditTime: 2021-05-08 09:47:26
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /Cyber-Security-ML-Toolbox/csmt/feature_selection/_bayes_fs.py
'''
import sys
sys.path.append("/Users/zhanghangsheng/Documents/my_code/Cyber-Security-ML-Toolbox")
from csmt.zoopt.bayes_opt.bayesian_optimization import BayesianOptimization
from csmt.classifiers.classic.logistic_regression import LogisticRegression
from csmt.classifiers.classic.decision_tree import DecisionTree
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
import numpy as np
def selectBayes(X,y):
    
    len_feature=X.shape[1]
    def score(my_array):
        for i in range(len_feature):
            my_array[i]=round(my_array[i])
        index=np.where(my_array==1)[0]
        X_=X[:,index]
        model=LogisticRegression() 
        score = cross_val_score(model.classifier, X_, y, cv=2).mean()  # 2次交叉验证
        return score
    # 实例化一个bayes优化对象
    bound=[]
    keys=[]
    for i in range(len_feature):
        bound.append([0,1])
        keys.append('my_array'+str(i))

    bo = BayesianOptimization(
        score,
        {'my_array':bound}
        )
    bo.maximize()
    print(bo.max['params'])
    max_x=np.array([bo.max['params'][key] for key in keys ])
    return max_x