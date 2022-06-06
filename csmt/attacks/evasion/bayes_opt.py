
import numpy as np

from csmt.zoopt.bayes_opt.bayesian_optimization import BayesianOptimization
from tqdm import tqdm
np.random.seed(10)
import time

class BayesOptMethod():

    def get_score(p):
        score=BayesOptMethod.estimator.predict(p.reshape(1,-1))
        return score[0][0]

    def __init__(self,estimator,eps,eps_step,max_iter,norm,upper,lower,feature_importance,mask):
        BayesOptMethod.estimator=estimator
        self.feature_importance=feature_importance
        self.norm=norm
        self.eps=eps
        self.eps_step=eps_step
        self.max_iter=max_iter
        self.upper=upper
        self.lower=lower
        self.mask=mask
        
    def _get_single_x(self,x):
        # start_time=time.time()
        bound=np.zeros((x.shape[0],2),dtype=float)
        keys=[]
        x_adv_path=np.zeros((1,2,x.shape[0]))
        x_adv_path[0,0]=x
        for i in range(x.shape[0]):
            bound[i]=np.array([-self.eps,self.eps])+x[i]
            bound=np.clip(bound, 0, 1)
            keys.append('x'+str(i))
        optimizer = BayesianOptimization(f=BayesOptMethod.get_score,pbounds={'x':bound},random_state=7,verbose=0)
        optimizer.maximize(init_points=5, n_iter=15)
        max_x=np.array([optimizer.max['params'][key] for key in keys])
        x_adv=max_x
        x_adv_path[0,1]=max_x
        # end_time=time.time()
        # print(end_time-start_time)
        return x_adv,x_adv_path
            
    def generate(self,X,y):
        X_adv_path=np.zeros((X.shape[0],2,X.shape[1]))
        X_size=X.shape[1]
        num=X.shape[0]
        X_adv=np.zeros(shape=(num,X_size))
        for i in tqdm(range(num)):
            x,x_adv_path=self._get_single_x(X[i])
            X_adv[i]=x
            X_adv_path[i]=x_adv_path
        y_adv=y
        return X_adv,y_adv,X_adv_path
