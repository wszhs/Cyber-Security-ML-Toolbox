import numpy as np
import sys
ROOT_PATH='/Users/zhanghangsheng/Documents/my_code/Cyber-Security-ML-Toolbox'
sys.path.append(ROOT_PATH)
from csmt.zoopt.gradient_free_optimizers import HillClimbingOptimizer


def convex_function(pos_new):
    score = -(pos_new["x1"] * pos_new["x1"] + pos_new["x2"] * pos_new["x2"])
    return score


search_space = {
    "x1": np.arange(-100, 101, 1),
    "x2": np.arange(-100, 101, 1),
}
opt = HillClimbingOptimizer(search_space)
opt.search(convex_function, n_iter=300)
