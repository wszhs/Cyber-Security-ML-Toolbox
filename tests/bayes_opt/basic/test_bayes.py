'''
Author: your name
Date: 2021-05-13 19:08:29
LastEditTime: 2021-05-13 19:08:42
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /Cyber-Security-ML-Toolbox/tests/bayes_opt/test_bayes.py
'''
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import numpy as np

# 目标函数                                                                                                                                                                  
objective = np.vectorize(lambda x, sigma_n=0: 0.001775 * x**5 - 0.055 * x**4 + 0.582 * x**3 - 2.405 * x**2 + 3.152 * x + 4.678 + np.random.normal(0, sigma_n))

# 采样函数 - GP-UCB                                                                                                                                                         
GPUCB = np.vectorize(lambda mu, sigma, t, ld, delta=0.1: mu + (1 * 2 * np.log(ld * t**2 * np.pi**2 / (6 * delta)))**0.5 * sigma)

# 超参数                                                                                                                                                                    
mean, l, sigma_f, sigma_n = 5, 1, 1, 0.0001

# 迭代次数                                                                                                                                                                  
max_iter = 3

# SE协方差函数                                                                                                                                                              
kernel = lambda r_2, l: np.exp(-r_2 / (2 * l**2))

# 初始训练样本，以一维输入为例                                                                                                                                              
X = np.arange(0.5, 10, 3.0)
X = X.reshape(X.size, 1)
Y = objective(X).flatten()

plt.figure(figsize=(8,5))

for i in range(max_iter):

    Xs = np.arange(0, 10, 0.1)
    Xs = Xs.reshape(Xs.size, 1)

    n, d = X.shape
    t = np.repeat(X.reshape(n, 1, d), n, axis=1) - X
    r_2 = np.sum(t**2, axis=2)
    Kf = sigma_f**2 * kernel(r_2, l)
    Ky = Kf + sigma_n**2 * np.identity(n)
    Ky_inv = np.linalg.inv(Ky)

    m = Xs.shape[0]
    t = np.repeat(Xs.reshape(m, 1, d), n, axis=1) - X
    r_2 = np.sum(t**2, axis=2).T
    kf = sigma_f**2 * kernel(r_2, l)

    mu = mean + kf.T @ Ky_inv @ (Y - mean)
    sigma = np.sqrt(sigma_f**2 - np.sum(kf.T @ Ky_inv * kf.T, axis=1))

    y_acf = GPUCB(mu, sigma, i + 1, n)
    sample_x = Xs[np.argmax(y_acf)]

    x_test = Xs.flatten()
    y_obj = objective(x_test).flatten()

    ax = plt.subplot(2, max_iter, i + 1)
    ax.set_title('t=%d' % (i + 1))
    plt.ylim(3, 8)
    plt.plot(x_test, mu, c='black', lw=1)
    plt.fill_between(x_test, mu + sigma, mu - sigma, alpha=0.2, color='#9FAEB2', lw=0)
    plt.plot(x_test, y_obj, c='red', ls='--', lw=1)
    plt.scatter(X, Y, c='red', marker='o', s=20)
    plt.subplot(2, max_iter, i + 1 + max_iter)
    plt.ylim(3.5, 9)
    plt.plot(x_test, y_acf, c='#18D766', lw=1)
    X = np.insert(X, 0, sample_x, axis=0)
    Y = np.insert(Y, 0, objective(sample_x))

plt.show()