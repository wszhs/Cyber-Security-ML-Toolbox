import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("/Users/zhanghangsheng/Documents/my_code/Cyber-Security-ML-Toolbox")
from csmt.zoopt.openbox import Optimizer, sp

# Define Search Space
space = sp.Space()
x1 = sp.Real("x1", 10, 250, default_value=100)
x2 = sp.Real("x2", 10, 20, default_value=15)
x3 = sp.Real("x3", 20, 30, default_value=25)
space.add_variables([x1, x2,x3])


def objective(x):
    print(x)
    print(x.keys())
    RMS=x['x1']**2+x['x2']*2 +x['x3']*4 # x is going to be a 20x1 array
    score=-1.0/(0.1+RMS)
    return score

opt = Optimizer(
    objective,
    space,
    max_runs=5,
    surrogate_type='gp',
    time_limit_per_trial=30,
    task_id='quick_start',
)
history = opt.run()

data=history.get_incumbents()

# history.plot_convergence()
# plt.show()