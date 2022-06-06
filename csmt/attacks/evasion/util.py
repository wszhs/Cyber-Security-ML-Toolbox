'''
Author: your name
Date: 2021-07-28 12:33:38
LastEditTime: 2021-07-28 12:33:39
LastEditors: your name
Description: In User Settings Edit
FilePath: /Cyber-Security-ML-Toolbox/csmt/attacks/evasion/util.py
'''
'''
Author: your name
Date: 2021-07-28 12:33:38
LastEditTime: 2021-07-28 12:33:38
LastEditors: your name
Description: In User Settings Edit
FilePath: /Cyber-Security-ML-Toolbox/csmt/attacks/evasion/util.py
'''
import numpy as np

def get_distribute(weight):
    len_weight=weight.shape[0]
    new_weight=np.zeros((len_weight))
    weight_all=0
    for i in range(len_weight):
        weight_all=weight_all+weight[i]
    for i in range(len_weight):
        new_weight[i]=weight[i]/weight_all
    return new_weight

def bisection(a, eps, xi, ub=1):
    pa = np.clip(a, 0, ub)
    if np.sum(pa) <= eps:
        # print('np.sum(pa) <= eps !!!!')
        w = pa
    else:
        mu_l = np.min(a - 1)
        mu_u = np.max(a)
        mu_a = (mu_u + mu_l)/2
        while np.abs(mu_u - mu_l) > xi:
            # print('|mu_u - mu_l|:',np.abs(mu_u - mu_l))
            mu_a = (mu_u + mu_l) / 2
            gu = np.sum(np.clip(a - mu_a, 0, ub)) - eps
            gu_l = np.sum(np.clip(a - mu_l, 0, ub)) - eps
            # print('gu:',gu)
            if gu == 0:
                # print('gu == 0 !!!!!')
                break
            if np.sign(gu) == np.sign(gu_l):
                mu_l = mu_a
            else:
                mu_u = mu_a

        w = np.clip(a - mu_a, 0, ub)

    return w
