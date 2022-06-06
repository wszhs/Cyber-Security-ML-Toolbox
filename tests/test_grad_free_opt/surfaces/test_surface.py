import numpy as np
import sys
ROOT_PATH='/Users/zhanghangsheng/Documents/my_code/Cyber-Security-ML-Toolbox'
sys.path.append(ROOT_PATH)
from csmt.zoopt.surfaces.test_functions import SphereFunction, AckleyFunction,RastriginFunction,RosenbrockFunction
from csmt.zoopt.surfaces.test_functions import BealeFunction,HimmelblausFunction,HölderTableFunction,CrossInTrayFunction
from csmt.zoopt.surfaces.test_functions import EasomFunction,BoothFunction,GoldsteinPriceFunction,StyblinskiTangFunction
from csmt.zoopt.surfaces.test_functions import McCormickFunction
from csmt.zoopt.surfaces.visualize import plotly_surface


step_ = 0.05
min_ = 10
max_ = 10
search_space = {
    "x0": np.arange(-min_, max_, step_),
    "x1": np.arange(-min_, max_, step_),
}
# function_arr=[BealeFunction(),HimmelblausFunction(),HölderTableFunction(),CrossInTrayFunction()]
function_arr=[EasomFunction(),BoothFunction(),GoldsteinPriceFunction(),StyblinskiTangFunction(2)]
# function_arr=[McCormickFunction(),SphereFunction(n_dim=2, metric="score"),AckleyFunction(metric="loss"),RastriginFunction(2)]

# rosenbrock_function = RosenbrockFunction()
# plotly_surface(rosenbrock_function,search_space).show()

for i in range(len(function_arr)):
    plotly_surface(function_arr[i],search_space).show()