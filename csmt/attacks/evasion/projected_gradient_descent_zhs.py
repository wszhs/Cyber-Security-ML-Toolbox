'''
Author: your name
Date: 2021-07-12 14:26:22
LastEditTime: 2021-07-27 14:05:08
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /Cyber-Security-ML-Toolbox/csmt/attacks/evasion/fast_gradient_zhs.py
'''
import numpy as np
from csmt.utils import (
    check_and_transform_label_format,
)
class PGDZhsMethod():
    def __init__(self,estimator=None,norm=np.inf,eps=0.3,eps_step=0.1,max_iter=100,upper=1,lower=0):
        self.estimator=estimator
        self.norm=norm
        self.eps=eps
        self.eps_step= eps_step
        self.max_iter=max_iter 
        self.upper=upper
        self.lower=lower
        
    def generate(self, x, y):
        y = check_and_transform_label_format(y, self.estimator.nb_classes)
        grad = np.zeros((x.shape[0],x.shape[1]))
        grads=np.zeros((self.max_iter,x.shape[0],x.shape[1]))
        x_adv_path=np.zeros((x.shape[0],self.max_iter+1,x.shape[1]))
        x_adv_path[:,0]=x
        for i in range(self.max_iter):
            cur_grad = self.estimator.loss_gradient(x+grad, y)
            if self.norm in [np.inf, "inf"]:
                grad=grad+(self.eps_step*np.sign(cur_grad))
                grad = np.clip(grad, -self.eps, self.eps)
                grad = np.clip(x+grad, self.lower, self.upper) - x
            grads[i,:]=grad
            x_adv_path[:,i+1]=x+grads[i,:]
        x_adv = x + grads[self.max_iter-1]
        return x_adv,x_adv_path