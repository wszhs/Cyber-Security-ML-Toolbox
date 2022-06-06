'''
Author: your name
Date: 2021-04-24 18:49:01
LastEditTime: 2021-04-24 18:53:53
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /Cyber-Security-ML-Toolbox/tests/test_Stackelberg/models.py
'''
import Genetic
import mcmc
import plot
import matplotlib
matplotlib.use('TkAgg')

l = 5
u = 12

if __name__ == '__main__':
    gen_size = 50
    q = Genetic.create_chromosome()  # q是基因上下标，value是基因对应的值
    value_xy = Genetic.creat_value(l, u)  # 生成xy的值
    low_dec = [[0, 0, 0] for i in range(gen_size)]
    upper_dec = [[0, 0, 0] for i in range(gen_size)]
    for i in range(gen_size):
        print("当前是第{}次博弈".format(i+1))
        islower = True
        value_xy = Genetic.gen_main(q, value_xy, l, u, gen_size, islower)
        tmp1 = [k for k in value_xy[0]]
        low_dec[i] = [k for k in value_xy[0]]
        for j in range(Genetic.pop_size):       # 下层根据对方x的取值，采取可能最优的策略，并将这种策略运用于每一种方案
            value_xy[j][1] = value_xy[0][1]    # 设置y值全部一样
        value_xy = Genetic.re_create_x(value_xy, l, u)     # 重置x
        islower = False
        value_xy = Genetic.gen_main(q, value_xy, l, u, gen_size, islower)
        tmp2 = [k for k in value_xy[0]]
        upper_dec[i] = [k for k in value_xy[0]]
        for j in range(Genetic.pop_size):    # 上层根据对方x的取值，采取可能最优的策略，并将这种策略运用于每一种方案
            value_xy[j][0] = value_xy[0][0]   # 设置x值全部一样
        print(low_dec)
        print(upper_dec)
        value_xy = Genetic.re_create_y(value_xy)
    print("下层的最优选择是：{}".format(tmp1))
    print("上层的最优选择是：{}".format(tmp2))
    p = plot.plot(low_dec, upper_dec)
    p.plot_xy()
    p.plots_3D()