'''
Author: your name
Date: 2021-04-24 18:50:58
LastEditTime: 2021-04-24 18:53:33
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /Cyber-Security-ML-Toolbox/tests/test_Stackelberg/plot.py
'''
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

class plot(object):
    def __init__(self, decision_low, decision_up):
        self.low_x = [low_dec[0] for low_dec in decision_low]  # 下层的x值
        self.low_y = [low_dec[1] for low_dec in decision_low]
        self.low_objective = [low_dec[2] for low_dec in decision_low]  # 下层的value值-期望
        self.upper_x = [upper_dec[0] for upper_dec in decision_up]  # 上层的x值
        self.upper_y = [upper_dec[1] for upper_dec in decision_up]  # 上层的y值
        self.upper_objective = [upper_dec[2] for upper_dec in decision_up]  # 上层的value值-期望

    def plots_3D(self):               # 在console里面运行正常，但是直接调用好像出错，原因未知
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for xs, ys, zs, color, shape in [(self.low_x, self.low_y, self.low_objective, 'r', 'o'),
                           (self.upper_x, self.upper_x, self.upper_objective, 'b', '^')]:
            ax.scatter(xs, ys, zs, c=color, marker=shape)
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        plt.show()
    def plot_xy(self):
        for s1, s2 in [(self.low_x, self.low_y),(self.upper_x, self.upper_y)]:
            fig, axs = plt.subplots(1, 1)
            plt.plot(s1, label='x')
            plt.plot(s2, label='y')
            # generate a legend box
            plt.legend()
            axs.set_xlim(0, 50)
            axs.set_ylim(2, 12)
            axs.set_xlabel('time')
            axs.set_ylabel('x and y')
            axs.grid(True)
            fig.tight_layout()

            plt.show()