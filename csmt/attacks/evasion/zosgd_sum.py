
import numpy as np
import copy
from csmt.attacks.evasion.abstract_evasion_sum import AbstractEvasionSum
np.random.seed(10)
class ZOSGDSumMethod(AbstractEvasionSum):
    def gradient_estimation_sum(self,mu,q,x,kappa,target_label,const,weights):
        sigma = 100
        grad_est=0
        d=x.shape[1]
        f_0,ignore=self.function_evaluation_cons_sum(x,kappa,target_label,const,x,weights)
        for i in range(q):
            u = np.random.normal(0, sigma, (1,d))
            u_norm = np.linalg.norm(u)
            u = u/u_norm
            f_tmp, ignore = self.function_evaluation_cons_sum(x+mu*u,kappa,target_label,const,x,weights)
            grad_est=grad_est+ (d/q)*u*(f_tmp-f_0)/(mu)
        return grad_est

