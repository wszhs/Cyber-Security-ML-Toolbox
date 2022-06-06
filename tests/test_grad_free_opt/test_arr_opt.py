from re import search
import numpy as np
import sys
ROOT_PATH='/Users/zhanghangsheng/Documents/my_code/Cyber-Security-ML-Toolbox'
sys.path.append(ROOT_PATH)
from csmt.zoopt.gradient_free_optimizers import HillClimbingOptimizer
from csmt.zoopt.gradient_free_optimizers  import BayesianOptimizer,RandomSearchOptimizer

def objective(x):
    p=np.zeros(len(x))
    for i in range(len(x)):
        p[i]=x['x'+str(i)]
    RMS=p[0]**2+p[1]*2 +p[2]*4 # x is going to be a 20x1 array
    score=1.0/(0.1+RMS)
    return score


search_space = {
    "x0": np.arange(10, 250, 1),
    "x1": np.arange(10, 20, 1),
    'x2': np.arange(20, 30, 1)
}
# opt = HillClimbingOptimizer(search_space)
# opt.search(objective, n_iter=3000)

# opt = RandomSearchOptimizer(search_space)
opt = HillClimbingOptimizer(search_space)
# opt = BayesianOptimizer(search_space)

opt.search(objective, n_iter=100,verbosity='progress_bar')
print(opt.best_para)
max_x=np.zeros(len(opt.best_para))
for i in range(len(opt.best_para)):
    max_x[i]=opt.best_para['x'+str(i)]
history=opt.score_l

print(history)

# search_space={}
# search_space.update({'x0':np.arange(10,250,1)})
# search_space.update({'x1':np.arange(10,250,1)})
# print(search_space)
