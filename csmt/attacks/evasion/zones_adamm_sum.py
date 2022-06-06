'''
Author: your name
Date: 2021-08-02 09:22:24
LastEditTime: 2021-08-04 10:03:19
LastEditors: your name
Description: In User Settings Edit
FilePath: /Cyber-Security-ML-Toolbox/csmt/attacks/evasion/zones_adamm_sum.py
'''
'''
Author: your name
Date: 2021-07-27 16:05:42
LastEditTime: 2021-07-28 10:12:06
LastEditors: your name
Description: In User Settings Edit
FilePath: /Cyber-Security-ML-Toolbox/csmt/attacks/evasion/zosgd_sum.py
'''
'''
Author: your name
Date: 2021-07-22 14:03:50
LastEditTime: 2021-07-27 17:09:47
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /Cyber-Security-ML-Toolbox/csmt/attacks/evasion/zones.py
'''
import numpy as np
import copy
from csmt.attacks.evasion.zoadamm_sum import ZOAdammSumMethod
np.random.seed(10)

class ZONESAdammSumMethod(ZOAdammSumMethod):

    def gradient_estimation_sum(self,mu,q,x,kappa,target_label,const,weights):
        sigma = 100
        q_prime=int(np.ceil(q/2))
        grad_est=0
        d=x.shape[1]
        for i in range(q_prime):
            u = np.random.normal(0, sigma, (1,d))
            u_norm = np.linalg.norm(u)
            u = u/u_norm
            f_tmp1, ignore = self.function_evaluation_cons_sum(x+mu*u,kappa,target_label,const,x,weights)
            f_tmp2, ignore = self.function_evaluation_cons_sum(x-mu*u,kappa,target_label,const,x,weights)
            grad_est=grad_est+ (d/q)*u*(f_tmp1-f_tmp2)/(2*mu)
        return grad_est

