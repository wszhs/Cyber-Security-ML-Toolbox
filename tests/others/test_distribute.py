'''
Author: your name
Date: 2021-04-10 10:56:10
LastEditTime: 2021-04-10 16:58:41
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /Cyber-Security-ML-Toolbox/tests/test_distr.py
'''
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
# from csmt.zoopt.DE import DE
from sko.DE import DE
import pandas as pd
import matplotlib.pyplot as plt
# num_evasion_attack=2
# evasion_distribute_arr=[]
# for i in range(num_evasion_attack):
    
def obj_func(p):
    x1, x2, x3 = p
    print(p)
    return x1 ** 2 + x2 ** 2 + x3 ** 2

constraint_eq = [
    lambda x: 1 - x[0]-x[1]
]
de = DE(func=obj_func, n_dim=3, size_pop=10, max_iter=10, lb=[0, 0, 0], ub=[5, 5, 5],
            constraint_eq=constraint_eq)

best_x, best_y = de.run()
print('best_x:', best_x, '\n', 'best_y:', best_y)

# Y_history = pd.DataFrame(de.all_history_Y)
# fig, ax = plt.subplots(2, 1)
# ax[0].plot(Y_history.index, Y_history.values, '.', color='red')
# Y_history.min(axis=1).cummin().plot(kind='line')
# plt.show()