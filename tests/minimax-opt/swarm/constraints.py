'''
Author: your name
Date: 2021-04-24 18:51:42
LastEditTime: 2021-04-24 18:51:43
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /Cyber-Security-ML-Toolbox/tests/test_Stackelberg/constrain.py
'''
import mcmc
import model


def meet_constrain_y(x, y):   # 下层约束
    return 1 if (mcmc.pr1(x, y) >= 0.99) and (mcmc.pr2(x, y) >= 0.99) and (y >= 0) else 0


def meet_constrain_x(x):              # 上层约束
    return 1 if (x >= model.l) and (x <= model.u) else 0