'''
Author: your name
Date: 2021-07-28 20:56:15
LastEditTime: 2021-08-04 09:58:24
LastEditors: your name
Description: In User Settings Edit
FilePath: /Cyber-Security-ML-Toolbox/csmt/attacks/evasion/zoadamm_sum.py
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
from csmt.attacks.evasion.abstract_evasion_sum import AbstractEvasionSum
from csmt.attacks.evasion.util import get_distribute

np.random.seed(10)
class ZOAdammSumMethod(AbstractEvasionSum):

    def generate_single(self,x,y):
        # print('next')
        beta_1=0.9
        beta_2=0.9999
        v_init=1e-7
        v_hat = v_init * np.ones((1,x.shape[1]))
        v = v_init * np.ones((1,x.shape[1]))
        const=self.init_const
        x_orig=copy.deepcopy(x)
        x_adv=copy.deepcopy(x)
        delta_adv = np.zeros((self.max_iter,1,x.shape[1]))
        m = np.zeros((1,x.shape[1]))
        len_models=len(self.estimator.model)
        weights=np.ones(len_models,dtype=np.float32)*1.0/len_models
        # weights=np.array([0,0,1])
        total_loss = np.zeros((self.max_iter,len_models))
        for i in range(0,self.max_iter):
            base_lr = self.eps_step/np.sqrt(i+1)
            # base_lr = self.eps_step
            grad_est=self.gradient_estimation_sum(self.mu,self.q,x,self.kappa,y,const,weights)
            if self.norm in [np.inf, "inf"]:
                m = beta_1 * m + (1-beta_1) * grad_est
                v = beta_2 * v + (1 - beta_2) * np.square(grad_est) ### vt
                v_hat = np.maximum(v_hat,v)
                delta_adv[i] = delta_adv[i-1] - base_lr * m /np.sqrt(v_hat)
                delta_adv[i]=np.clip(delta_adv[i],-self.eps,self.eps)

            total_loss[i]=self.function_evaluation_cons_models(x_orig+delta_adv[i],y,weights)
            w_grad = total_loss[i] - 2 * self.lmd * (weights-1/(len_models))
            w_proj = weights + self.beta* w_grad
            weights = get_distribute(w_proj)
            # print(weights)
        
        for i in range(0,self.max_iter):
            x_adv_tmp=x_orig+delta_adv[i]
            orig_prob = self.estimator.predict(x_adv_tmp)
            predict_arr=np.zeros(len_models)
            for j in range(len_models):
                label_=np.argmax(orig_prob[j], axis=1)
                if label_ !=y:
                    predict_arr[j]=True
            # print(predict_arr)
                    
        x_adv = x_adv+delta_adv[self.max_iter-1]
        return x_adv


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

