'''
Author: your name
Date: 2021-07-23 10:16:08
LastEditTime: 2021-08-02 10:49:12
LastEditors: your name
Description: In User Settings Edit
FilePath: /Cyber-Security-ML-Toolbox/csmt/attacks/evasion/zoadamm.py
'''
'''
Author: your name
Date: 2021-07-23 10:16:08
LastEditTime: 2021-07-28 20:35:19
LastEditors: your name
Description: In User Settings Edit
FilePath: /Cyber-Security-ML-Toolbox/csmt/attacks/evasion/zoadamm.py
'''
'''
Author: your name
Date: 2021-07-22 14:03:50
LastEditTime: 2021-07-27 14:44:25
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /Cyber-Security-ML-Toolbox/csmt/attacks/evasion/zones.py
'''
import numpy as np
import copy
from csmt.attacks.evasion.abstract_evasion import AbstractEvasion

np.random.seed(10)
class ZOAdaMMMethod(AbstractEvasion):

    def generate_single(self,x,y):
        const=self.init_const
        x_adv=copy.deepcopy(x)
        x_adv_tmp=copy.deepcopy(x_adv)
        beta_1=0.9
        beta_2=0.99
        v_init = 1e-7 #0.00001
        # v_hat = v_init * np.ones((1,x.shape[1]))
        v = v_init * np.ones((1,x.shape[1]))
        delta_adv = np.zeros((1,self.max_iter,x.shape[1]))
        x_adv_path=np.zeros((1,self.max_iter+1,x.shape[1]))
        x_adv_path[0,0]=x
        m = np.zeros((1,x.shape[1]))
        for i in range(0,self.max_iter):
            # base_lr = self.eps_step/np.sqrt(i+1)
            base_lr = self.eps_step
            grad_est=self.gradient_estimation(self.mu,self.q,x_adv_tmp,self.kappa,y,const)
            if self.norm in [np.inf, "inf"]:
                m = beta_1 * m + (1-beta_1) * grad_est
                v = beta_2 * v + (1 - beta_2) * np.square(grad_est) ### vt
                # v_hat = np.maximum(v_hat,v)
                delta_adv[0,i] = delta_adv[0,i-1] - base_lr * m /np.sqrt(v)
                if self.mask is not None:
                    delta_adv[0,i] = np.where(self.mask == 0.0, 0.0, delta_adv[0,i])
                delta_adv[0,i]=np.clip(delta_adv[0,i],-self.eps,self.eps)
                delta_adv[0,i] = np.clip(x_adv+delta_adv[0,i], 0.0, 1.0) - x_adv
            x_adv_tmp=x+delta_adv[0,i]
            x_adv_path[0,i+1]=x+delta_adv[0,i]

        x_adv = x + delta_adv[0,self.max_iter-1]
        return x_adv,x_adv_path

    def gradient_estimation(self,mu,q,x,kappa,target_label,const):
        sigma = 100
        grad_est=0
        d=x.shape[1]
        f_0,ignore=self.function_evaluation_cons(x,kappa,target_label,const,x)
        for i in range(q):
            u = np.random.normal(0, sigma, (1,d))
            u_norm = np.linalg.norm(u)
            u = u/u_norm
            f_tmp, ignore = self.function_evaluation_cons(x+mu*u,kappa,target_label,const,x)
            grad_est=grad_est+ (d/q)*u*(f_tmp-f_0)/(mu)
        return grad_est
