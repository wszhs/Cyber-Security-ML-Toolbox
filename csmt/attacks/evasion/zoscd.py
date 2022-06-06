
import numpy as np
import copy

from numpy.core.fromnumeric import size
from csmt.attacks.evasion.abstract_evasion import AbstractEvasion

np.random.seed(10)
class ZOSCDMethod(AbstractEvasion):

    def gradient_estimation(self,mu,q,x,kappa,target_label,const):
        grad_est=0
        d=x.shape[1]
        idx_coords_random=np.random.randint(d,size=q)
        for id_coord in range(q):
            idx_coord=idx_coords_random[id_coord]
            u = np.zeros(d)
            u[idx_coord]=1
            u=np.resize(u,x.shape)
            f_old, ignore = self.function_evaluation_cons(x-mu*u,kappa,target_label,const,x)
            f_new, ignore = self.function_evaluation_cons(x+mu*u,kappa,target_label,const,x)
            grad_est=grad_est+ (d/q)*u*(f_new-f_old)/(2*mu)
        # print(grad_est)
        return grad_est

