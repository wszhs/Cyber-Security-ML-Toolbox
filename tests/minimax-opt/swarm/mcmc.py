'''
Author: your name
Date: 2021-04-24 18:50:37
LastEditTime: 2021-04-24 18:50:37
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /Cyber-Security-ML-Toolbox/tests/test_Stackelberg/mcmc.py
'''
import random
import math

A1 = 30
A2 = 10
B1 = 22
B2 = 8

aerf1 = 0.99
aerf2 = 0.99

M = 10000      #蒙特卡洛模拟生成随机数个数
eta1 = [random.uniform(2, 4) for i in range(M)]
eta2 = [4 * math.exp(-4 * random.uniform(2, 4)) for i in range(M)]
lamda1 = [random.uniform(3, 4) for i in range(M)]
lamda2 = [3 * math.exp(-3 * random.uniform(3, 4)) for i in range(M)]


def low_expection(index):       # 上层期望值
    fx = []
    for xy in index:
        x = xy[0]
        y = xy[1]
        sum_low = 0
        for i in range(M):
            sum_low = sum_low + (B1 + lamda1[i] - x - 2 * y) * y - (y - B2 - lamda2[i]) ** 2
        fx.append(sum_low/M)
    return fx


def upper_expection(index):            # 下层期望值
    fx = []
    for xy in index:
        x = xy[0]
        y = xy[1]
        sum_upper = 0
        for i in range(M):
            sum_upper = sum_upper + (A1 + eta1[i] - 2 * x - y) * x - (x - A2 - eta2[i]) ** 2
        fx.append(sum_upper / M)
    return fx


def pr1(x, y):           #第一个概率约束
    count = 0
    for i in range(M):
        if 2 * x + y <= A1 + eta1[i]:
            count += 1
    pro = count/M
    return pro


def pr2(x, y):                # 第二个概率约束
    count = 0
    for i in range(M):
        if x + 2 * y <= B1 + lamda1[i]:
            count += 1
    pro = count / M
    return pro
