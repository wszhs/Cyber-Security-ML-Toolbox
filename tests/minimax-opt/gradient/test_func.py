
import numpy as np


def obj_1(x):
    dim = 2
    x_opt = 0.5
    is_noise = False
    n = dim // 2 - 1
    x_ne = np.array([x_opt] * dim)
    val = 0
    for i, _x in enumerate(x):
        if i > n:
            val = val - (_x - x_ne[i]) ** 2
        else:
            val = val + (_x - x_ne[i]) ** 2
    return val + 0.025 * is_noise * np.random.randn()

print(obj_1(x))