'''
Author: your name
Date: 2021-04-10 11:08:41
LastEditTime: 2021-05-15 16:19:32
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /Cyber-Security-ML-Toolbox/tests/test_de.py
'''
'''
min f(x1, x2, x3) = x1^2 + x2^2 + x3^2
s.t.
    x1*x2 >= 1
    x1*x2 <= 5
    x2 + x3 = 1
    0 <= x1, x2, x3 <= 5
'''


import sys
sys.path.append("/Users/zhanghangsheng/Documents/my_code/Cyber-Security-ML-Toolbox")
# import os
# print(os.getcwd())
import matplotlib
matplotlib.use('TkAgg')
from csmt.zoopt.DE import DE
import pandas as pd
import matplotlib.pyplot as plt

def obj_func(p):
    x1, x2, x3 = p
    return x1 ** 2 + x2 ** 2 + x3 ** 2

constraint_eq = [
    lambda x: 1 - x[1] - x[2]
]

constraint_ueq = [
    lambda x: 1 - x[0] * x[1],
    lambda x: x[0] * x[1] - 5
]

if __name__=='__main__':
    de = DE(func=obj_func, n_dim=3, size_pop=50, max_iter=800, lb=[0, 0, 0], ub=[5, 5, 5],
            constraint_eq=constraint_eq, constraint_ueq=constraint_ueq)

    best_x, best_y = de.run()
    print('best_x:', best_x, '\n', 'best_y:', best_y)

    Y_history = pd.DataFrame(de.all_history_Y)
    # print(Y_history)
    
    fig, ax = plt.subplots(1, 1)
    # ax[0].plot(Y_history.index, Y_history.values, '.', color='red')
    # print(Y_history.min(axis=1))
    # Y_history.min(axis=1).cummin().plot(kind='line')
    Y_history.min(axis=1).cummin().plot(kind='line')
    plt.show()