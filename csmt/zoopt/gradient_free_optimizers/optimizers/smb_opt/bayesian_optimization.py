# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


from .exp_imp_based_opt import ExpectedImprovementBasedOptimization

from .surrogate_models import (
    GPR_linear,
    GPR,
    RandomForestRegressor
)

gaussian_process = {"gp_nonlinear": GPR(), "gp_linear": GPR_linear(), "rf": RandomForestRegressor()}


class BayesianOptimizer(ExpectedImprovementBasedOptimization):
    name = "Bayesian Optimization"
    _name_ = "bayesian_optimization"

    def __init__(
        self,
        *args,
        gpr=gaussian_process["gp_nonlinear"],
        xi=0.03,
        warm_start_smbo=None,
        max_sample_size=10000000,
        sampling={"random": 1000000},
        warnings=100000000,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.gpr = gpr
        self.regr = gpr
        self.xi = xi
        self.warm_start_smbo = warm_start_smbo
        self.max_sample_size = max_sample_size
        self.sampling = sampling
        self.warnings = warnings

        self.init_warm_start_smbo()
