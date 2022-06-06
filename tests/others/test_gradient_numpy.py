'''
Author: your name
Date: 2021-04-08 19:23:15
LastEditTime: 2021-04-08 19:25:40
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /Cyber-Security-ML-Toolbox/tests/test_gradient_numpy.py
'''

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def predict(x, a, b):
    return a * x + b

def loss(y, y_):
    v = y - y_
    return np.sum(v * v) / len(y)
# 手动对a求偏导
def partial_a(x, y, y_):
    return 2 / len(y) * np.sum((y-y_)*x)
# 手动对b求偏导
def partial_b(x, y, y_):
    return 2 / len(y) * np.sum(y-y_)
# 学习率
learning_rate = 0.0001
# 初始化参数a, b
a, b = np.random.normal(size=2)
x = np.arange(30)
# 这些数据大致在 y = 6x + 1 附近
y_ = [7.1, 4.3, 6.5, 28.2, 11.8, 40.2, 24.8, 56.1, 
     36.9, 53.0, 52.2, 57.1, 62.5, 79.7, 95.8, 83.6, 
     103.0, 104.7, 108.2, 116.5, 115.1, 121.2, 129.8, 
     148.1, 142.1, 151.5, 165.8, 174.7, 154.5, 189.9]
loss_list = []
plt.scatter(x, y_, color="green")
for i in range(60):
    y = predict(x, a, b)    
    lss = loss(y, y_)
    loss_list.append(lss)
    if i % 10 == 0:
        print("%03d weight a=%.2f, b=%.2f loss=%.2f" % (i, a, b, lss))
        plt.plot(x, predict(x, a, b), linestyle='--', label="epoch=%s" % i)
    # 采用梯度下降算法更新权重
    a = a - learning_rate * partial_a(x, y, y_)
    b = b - learning_rate * partial_b(x, y, y_)
print("final weight a=%.2f, b=%.2f loss=%.2f" % (a, b, lss))

plt.plot(x, predict(x, a, b), label="final")
plt.legend()
plt.figure()
plt.plot(loss_list, color='red', label="loss")
plt.legend()