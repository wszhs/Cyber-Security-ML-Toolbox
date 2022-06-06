'''
Author: your name
Date: 2021-04-01 17:30:49
LastEditTime: 2021-08-04 09:16:22
LastEditors: your name
Description: In User Settings Edit
FilePath: /Cyber-Security-ML-Toolbox/csmt/attacks/evasion/abstract_evasion.py
'''
import copy
import numpy as np

class AbstractEvasion:

    def __init__(self,estimator,eps,eps_step,max_iter,norm,upper,lower,feature_importance,mask):
        self.estimator=estimator
        self.feature_importance=feature_importance
        self.norm=norm
        self.eps=eps
        self.eps_step=eps_step
        self.max_iter=max_iter
        self.upper=upper
        self.lower=lower
        self.q = 20 ### number of random direction vectors
        self.mu = 0.05 ### key parameter: smoothing parameter in ZO gradient estimation # 0.001 for imagenet
        self.kappa = 1e-10
        self.init_const = 1 ### regularization parameter prior to attack loss
        self.mask=mask

    def generate(self, X, y):
        print(self.mask)
        X_adv=copy.deepcopy(X)
        X_adv_path=np.zeros((X_adv.shape[0],self.max_iter+1,X_adv.shape[1]))
        for i in range(X.shape[0]):
            # if i%10==0:
            #     print(i)
            x_singe=X[i:i+1]
            y_single=y[i:i+1]
            x_adv_single,x_adv_path=self.generate_single(x_singe,y_single)
            X_adv[i]=x_adv_single
            X_adv_path[i]=x_adv_path
        y_adv=y
        return X_adv,y_adv,X_adv_path

    def generate_single(self,x,y):
        const=self.init_const
        x_orig=copy.deepcopy(x)
        x_adv=copy.deepcopy(x)
        x_adv_tmp=copy.deepcopy(x_orig)
        delta_adv = np.zeros((1,self.max_iter,x.shape[1]))
        x_adv_path=np.zeros((1,self.max_iter+1,x.shape[1]))
        x_adv_path[0,0]=x
        iter=self.max_iter-1
        min_inter=np.inf
        for i in range(0,self.max_iter):
            # base_lr = self.eps_step/np.sqrt(i+1)
            base_lr = self.eps_step
            grad_est=self.gradient_estimation(self.mu,self.q,x_adv_tmp,self.kappa,y,const)
            if self.norm in [np.inf, "inf"]:
                delta_adv[0,i] =delta_adv[0,i-1]-base_lr*np.sign(grad_est)
                if self.mask is not None:
                    delta_adv[0,i] = np.where(self.mask == 0.0, 0.0, delta_adv[0,i])
                # delta_adv[0,i] =delta_adv[0,i-1]-base_lr*grad_est
                delta_adv[0,i]=np.clip(delta_adv[0,i],-self.eps,self.eps)
                delta_adv[0,i] = np.clip(x_adv+delta_adv[0,i], self.lower, self.upper) - x_adv
            x_adv_path[0,i+1]=x+delta_adv[0,i]
            x_adv_tmp=x_orig+delta_adv[0,i]
            # print(self.estimator.predict(x_adv_tmp)[0,y])

            if self.estimator.predict(x_adv_tmp)[0,y]<self.estimator.predict(x_adv_tmp)[0,0]:
                iter=i
                break

        x_adv = x + delta_adv[0,iter]
        return x_adv,x_adv_path

    def function_evaluation_cons(self,x_adv, kappa, target_label, const,x):
        orig_prob = self.estimator.predict(x_adv)
        tmp = orig_prob.copy()
        tmp[0, target_label] = 0
        Loss1=orig_prob[0, target_label] 
        Loss2 = np.linalg.norm(x_adv - x) ** 2 ### squared norm
        return Loss1, Loss2

    def gradient_estimation(self,mu,q,x,kappa,target_label,const):
        pass