'''
Author: your name
Date: 2021-04-01 17:45:01
LastEditTime: 2021-06-07 17:18:31
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /Cyber-Security-ML-Toolbox/csmt/attacks/evasion/de.py
'''

import sys

from numpy.core.records import array
from csmt.zoopt.DE import DE
from tqdm import tqdm
import numpy as np
import time 
    
class DEEvasionAttack():
    estimator=None
    count=0
    def get_score(p):
        DEEvasionAttack.count=DEEvasionAttack.count+1
        # print(DEEvasionAttack.count)
        score=DEEvasionAttack.estimator.predict(p.reshape(1,-1))
        return -score[0][0]

    def __init__(self,estimator,eps,eps_step,max_iter,norm,upper,lower,feature_importance,mask):
        DEEvasionAttack.estimator=estimator
        self.feature_importance=feature_importance
        self.norm=norm
        self.eps=eps
        self.eps_step=eps_step
        self.max_iter=max_iter
        self.upper=upper
        self.lower=lower
        self.mask=mask
        
    def _get_single_x(self,x,chrom_length):
        # start_time=time.time()
        bound=np.zeros((x.shape[0],2),dtype=float)
        x_adv_path=np.zeros((1,2,x.shape[0]))
        x_adv_path[0,0]=x
        for i in range(x.shape[0]):
            bound[i]=np.array([-self.eps,self.eps])+x[i]
            bound=np.clip(bound, 0, 1)
        lb=bound[:,0]
        ub=bound[:,1]
        de = DE(func=DEEvasionAttack.get_score, n_dim=chrom_length, size_pop=10, max_iter=50, lb=lb, ub=ub)
        best_x, best_y = de.run()
        x_adv_path[0,1]=best_x
        # end_time=time.time()
        # print(end_time-start_time)
        return best_x,x_adv_path
            
    def generate(self,X,y):
        X_adv_path=np.zeros((X.shape[0],2,X.shape[1]))
        X_size=X.shape[1]
        num=X.shape[0]
        X_adv=np.zeros(shape=(num,X_size))
        for i in tqdm(range(num)):
            DEEvasionAttack.count=0
            x,x_adv_path=self._get_single_x(X[i],X_size)
            X_adv[i]=x
            X_adv_path[i]=x_adv_path
        y_adv=y
        return X_adv,y_adv,X_adv_path




