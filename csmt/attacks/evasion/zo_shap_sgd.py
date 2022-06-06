
import numpy as np
import copy
from csmt.attacks.evasion.abstract_evasion import AbstractEvasion

np.random.seed(10)
class ZOShapSGDMethod(AbstractEvasion):

    def gradient_estimation(self,mu,q,x,kappa,target_label,const):
        sigma = 100
        grad_est=0
        d=x.shape[1]
        f_0,ignore=self.function_evaluation_cons(x,kappa,target_label,const,x)
        for i in range(q):
            import_index=np.array([0]) 
            # import_index=np.array([self.feature_importance]) 

            len_import=import_index.shape[0]

            u_import = np.random.normal(0, sigma, (1,len_import))
            u_norm_import = np.linalg.norm(u_import)
            u_import = u_import/u_norm_import

            u=np.zeros(d)
            u[import_index]=u_import

            f_tmp, ignore = self.function_evaluation_cons(x+mu*u,kappa,target_label,const,x)
            grad_est=grad_est+ (d/q)*u*(f_tmp-f_0)/(mu)
        return grad_est
