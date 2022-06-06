import numpy as np
def f1(z):
    x = z[0]
    y = z[1]
    f = -3*x*x-y*y+4*x*y
    return f

def f2(z):
    x = z[0]
    y = z[1]
    f = 3*x*x+y*y+4*x*y
    return f

def f3(z):
    x = z[0]
    y = z[1]
    f = (0.4*x*x-0.1*(y-3*x+0.05*x*x*x)**2-0.01*y*y*y*y)*np.exp(-0.01*(x*x+y*y))
    return f

def saddle(z):
    dim = 2
    x_opt = 0.5
    n = dim // 2 - 1
    x_ne = np.array([x_opt] * dim)
    f = 0.0
    for i, _x in enumerate(z):
        if i > n:
            f = f - (_x - x_ne[i]) ** 2
        else:
            f = f + (_x - x_ne[i]) ** 2
    return f

def mop1(x):

    f = (x[1] - 5.1 * (x[0] / (2. * np.pi)) ** 2 + (5. / np.pi) * x[0] - 6.) ** 2 + 10 * (
        (1 - (1. / (8. * np.pi))) * np.cos(x[0]) + 1.)
    return f