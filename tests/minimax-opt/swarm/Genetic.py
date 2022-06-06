'''
Author: your name
Date: 2021-04-24 18:49:43
LastEditTime: 2021-04-24 18:50:13
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /Cyber-Security-ML-Toolbox/tests/test_Stackelberg/genetic.py
'''
import random
import mcmc
import constraints
import numpy as np


pop_size = 50
evaluate_rate = 0.05
sel_cross_rank = 0.2
mutation_rank = 0.2


class gene(object):
    def __init__(self, low, upper, value):
        self.v_low = low
        self.v_upper = upper
        self.v_value = value


def rank(input_xy, is_low):
    object_f = mcmc.low_expection(input_xy) if is_low else mcmc.upper_expection(input_xy)
    input_xy = np.array(input_xy)
    object_f = np.insert(input_xy, 2, object_f, axis=1)   #插入第三列
    input_xy = object_f[np.lexsort(-object_f.T)]  # 对第二，第三列进行排序
    return input_xy  # 返回三列的数组，第一列是x的值，第二列是y的值，第三列是目标函数的值，并按照目标函数的值降序排列


def create_chromosome():
    q = [0] * (pop_size + 1)
    for i in range(1, pop_size + 1):  # +1是为了让最大值为30
        q[i] = q[i - 1] + evaluate_rate * pow(1 - evaluate_rate, i)  # q从1开始,到30为止。q0用默认值0
    return q


def creat_value(l, u):          # 创建x，y初始值
    value = [[0, 0] for i in range(pop_size)]
    # 生成随机数对应每一段基因
    for i in range(pop_size):
        x = random.uniform(l, u)
        while True:
            y = random.uniform(0, 100)
            if constraints.meet_constrain_y(x, y) == 1:
                value[i] = [x, y]
                break
    # value.sort(reverse=False)  # 对随机数从大到小进行排序
    return value


def re_create_x(value, l, u):         # 已知下层采取y的策略后，x更新
    for i in range(pop_size):
        x = random.uniform(l, u)
        value[i][0] = x
    return value


def re_create_y(value):
    for i in range(pop_size):
        x = value[i][0]
        while True:
            y = random.uniform(0, 100)
            if constraints.meet_constrain_y(x, y) == 1:
                value[i][1] = y
                break
    return value


def select(q, value):
    value_new = np.array([[0.0, 0.0] for i in range(pop_size)])
    for i in range(pop_size):  # 选择pop_size个基因
        r = random.uniform(0, q[pop_size])
        for j in range(pop_size):       # 判断落在基因那个段内
            if ((r >= q[j]) and (r < q[j + 1]) and (j != pop_size)) or ((r == q[j + 1]) and (j == pop_size)):
                # j做有下界无上界的匹配，最后一个基因包含上下界。或者是找到最后一个
                value_new[i] = value[j]
                break  # 找到了对应基因，随机生成下一个r
    return value_new               # 返回的是popsize个值，选中的值


def cross(value, islow):
    flag = 0  # 表示是否已经有一条等待被交叉的基因了
    temp_gen = 0  # 用于临时存储待交叉的基因,表示第n条基因
    for i, j in enumerate(value):  # i是下标(序号从零开始)，j是值
        if random.random() < sel_cross_rank:
            if flag == 0:
                temp_gen = i
                flag = 1  # 等待被交叉
            else:  # 进行交叉
                c = random.random()  # 取整值概率无限情况下为0，所以可以看成是开区间
                x = value[temp_gen][int(islow)]            # 只调整y值或者x值，下层交叉y值时islow为1
                y = j[int(islow)]
                value[temp_gen][int(islow)] = c * x + (1 - c) * y
                value[i][int(islow)] = (1 - c) * x + c * y
                flag = 0  # 一组已经完成交叉
    return value


def mutation(value, l, u, islow):              # 变异
    for i, num in enumerate(value):
        # print(num)
        if random.random() < mutation_rank:
            turn = 0
            x = num[0]
            y = num[1]
            if islow:             # 只对y进行变异
                for j in range(1000):
                    y = y + 10 * random.uniform(-1, 1)
                    if constraints.meet_constrain_y(x, y) == 1:  # 下层
                        value[i][1] = y
                        break
            else:               # 只对x进行变异
                for j in range(1000):
                    x = x + 5 * random.uniform(-1, 1)
                    if constraints.meet_constrain_x(x) == 1:
                        value[i][0] = x
                        break
    return value


def gen_main(q, value_xy, x_l, x_u, Gen, islow):  # 遗传算法，x取值的上界和下界
    for i in range(Gen):
        value_xy = rank(np.array(value_xy)[:, [0, 1]], islow)              # 计算适应度并排序
        # print(value_xy)
        value_xy = select(q, np.array(value_xy)[:, [0, 1]])           # 选择,返回选中的值
        value_xy = cross(value_xy, islow)                 # 交叉
        # print(value_xy)
        value_xy = mutation(value_xy, x_l, x_u, islow)     # 变异
        print("正在运行{}层，已经运行了{}%".format("下" if islow else "上", i * 100 / Gen))
    return rank(value_xy, islow)
