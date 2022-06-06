'''
Author: your name
Date: 2021-07-22 14:03:50
LastEditTime: 2021-08-03 14:25:20
LastEditors: your name
Description: In User Settings Edit
FilePath: /Cyber-Security-ML-Toolbox/csmt/attacks/evasion/zones.py
'''
'''
Author: your name
Date: 2021-07-22 14:03:50
LastEditTime: 2021-07-28 20:44:41
LastEditors: your name
Description: In User Settings Edit
FilePath: /Cyber-Security-ML-Toolbox/csmt/attacks/evasion/zones.py
'''
'''
Author: your name
Date: 2021-07-22 14:03:50
LastEditTime: 2021-07-27 14:41:19
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /Cyber-Security-ML-Toolbox/csmt/attacks/evasion/zones.py
'''
import numpy as np
import copy
from csmt.attacks.evasion.abstract_evasion import AbstractEvasion


np.random.seed(10)
class ZONESMethod(AbstractEvasion):

    def gradient_estimation(self,mu,q,x,kappa,target_label,const):
        sigma = 100
        q_prime = int(np.ceil(q/2))
        grad_est=0
        d=x.shape[1]
        for i in range(q_prime):
            u = np.random.normal(0, sigma, (1,d))
            u_norm = np.linalg.norm(u)
            u = u/u_norm
            f_tmp1, ignore = self.function_evaluation_cons(x+mu*u,kappa,target_label,const,x)
            f_tmp2, ignore = self.function_evaluation_cons(x-mu*u,kappa,target_label,const,x)
            grad_est=grad_est+ (d/q)*u*(f_tmp1-f_tmp2)/(2*mu)
        return grad_est
