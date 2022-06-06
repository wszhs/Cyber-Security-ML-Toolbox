'''
Author: your name
Date: 2021-04-21 19:09:27
LastEditTime: 2021-04-22 10:23:26
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /Cyber-Security-ML-Toolbox/tests/test_bayes_array.py
'''
import sys
sys.path.append("/Users/zhanghangsheng/Documents/my_code/Cyber-Security-ML-Toolbox")
from csmt.zoopt.bayes_opt.bayesian_optimization import BayesianOptimization
# from bayes_opt import BayesianOptimization
import numpy as np


def objective(x):
    RMS=x[0]**2+x[1]*2 +x[2]*4 # x is going to be a 20x1 array
    FF=1.0/(0.1+RMS)
    return FF

arr=np.array([[10, 250],[10,20],[20,30]])
optimizer = BayesianOptimization(
    objective,
    {'x':arr}
)

optimizer.maximize()
print(optimizer.max)

# def objective(x0,x1):
#     RMS=x0**2+x1*2  # x is going to be a 20x1 array
#     FF=1.0/(0.1+RMS)
#     return FF

# optimizer = BayesianOptimization(
#     objective,
#     {'x0':(10, 250),'x1':(10,20)}
# )

# optimizer.maximize()
# print(optimizer.max)

