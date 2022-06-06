'''
Author: your name
Date: 2021-05-13 18:55:25
LastEditTime: 2021-05-13 18:56:44
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /Cyber-Security-ML-Toolbox/tests/test_bayes_opt.py
'''
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import numpy as np

# SE协方差函数
kernel_se = np.vectorize(lambda x1, x2, l: np.exp(-(x1 - x2) ** 2 / (2 * l ** 2)))

def sample_se(x, l, mean=0):
    # x为numpy数组，e.g. x = np.arange(-5, 5, 0.05)
    x1, x2 = np.meshgrid(x, x)
    n = len(x)
    sigma = kernel_se(x1, x2, l) + np.identity(n) * 0.000000001
    L = np.linalg.cholesky(sigma)
    u = np.random.randn(n)
    y = mean + L @ u
    return y

c = ['red', 'green', 'blue']
l = [3, 1, 0.3]

for i in range(len(l)):
    x = np.arange(-5, 5, 0.05)
    y = sample_se(x, l[i])
    plt.plot(x, y, c=c[i], linewidth=1, label='l=%.1f' % l[i])

plt.xlabel('input, x')
plt.ylabel('output, f(x)')
plt.legend(loc='best')
plt.show()