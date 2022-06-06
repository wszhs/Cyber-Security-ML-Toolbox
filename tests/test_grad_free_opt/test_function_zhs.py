import numpy as np
import sys
ROOT_PATH='/Users/zhanghangsheng/Documents/my_code/Cyber-Security-ML-Toolbox'
sys.path.append(ROOT_PATH)
from csmt.zoopt.gradient_free_optimizers  import RandomSearchOptimizer,BayesianOptimizer,HillClimbingOptimizer,StochasticHillClimbingOptimizer,SimulatedAnnealingOptimizer,GridSearchOptimizer,ParticleSwarmOptimizer,EvolutionStrategyOptimizer,OneDimensionalBayesianOptimization
from csmt.zoopt.gradient_free_optimizers import TreeStructuredParzenEstimators,ForestOptimizer
from csmt.zoopt.surfaces.test_functions import SphereFunction, AckleyFunction,RastriginFunction,EasomFunction,GoldsteinPriceFunction

def obj_function(params):
    # score=SphereFunction(n_dim=2, metric="score").objective_function_dict(params)
    score=AckleyFunction(metric="loss").objective_function_dict(params)
    # score=EasomFunction().objective_function_dict(params)
    # score=GoldsteinPriceFunction().objective_function_dict(params)
    return score


search_space = {
    "x0": np.arange(-10, 10, 1),
    "x1": np.arange(-10, 10, 1),
}

# opt = RandomSearchOptimizer(search_space)
# opt=BayesianOptimizer(search_space)
opt=ForestOptimizer(search_space)
# opt=TreeStructuredParzenEstimators(search_space)
# opt=HillClimbingOptimizer(search_space)
# opt=StochasticHillClimbingOptimizer(search_space)
# opt=GridSearchOptimizer(search_space)
# opt=ParticleSwarmOptimizer(search_space)
# opt=EvolutionStrategyOptimizer(search_space)
# opt=OneDimensionalBayesianOptimization(search_space)
opt.search(obj_function, n_iter=200)

history=opt.score_l
history=np.array(history)

print(history)
