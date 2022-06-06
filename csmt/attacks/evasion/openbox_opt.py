
import numpy as np

from csmt.zoopt.openbox import Optimizer, sp
from tqdm import tqdm
np.random.seed(10)
import time

class OpenboxMethod():
    estimator=None
    count=0

    def get_score(x):
        OpenboxMethod.count=OpenboxMethod.count+1
        # print(OpenboxMethod.count)
        len_x=len(x.keys())
        p=np.zeros(len_x)
        for i in range(len_x):
            p[i]=x[x.keys()[i]]
        score=OpenboxMethod.estimator.predict(p.reshape(1,-1))
        return -score[0][0]

    def __init__(self,estimator,eps,eps_step,max_iter,norm,upper,lower,feature_importance,mask):
        OpenboxMethod.estimator=estimator
        self.feature_importance=feature_importance
        self.norm=norm
        self.eps=eps
        self.eps_step=eps_step
        self.max_iter=max_iter
        self.upper=upper
        self.lower=lower
        self.mask=mask
        
    def _get_single_x(self,x):

        x_adv_path=np.zeros((1,2,x.shape[0]))
        x_adv_path[0,0]=x

        # Define Search Space
        space = sp.Space()
        x_space=[]
        for i in range(x.shape[0]):
            bound=np.array([-self.eps,self.eps])+x[i]
            bound=np.clip(bound, 0, 1)
            x_s = sp.Real('x'+str(i), bound[0], bound[1], default_value=x[i])
            x_space.append(x_s)
        space.add_variables(x_space)

        opt = Optimizer(
            OpenboxMethod.get_score,
            space,
            max_runs=5,
            surrogate_type='gp',
            time_limit_per_trial=30,
            task_id='quick_start',
        )

        history = opt.run()
        data=history.get_incumbents()[0][0]

        max_x=np.zeros(len(x))
        for i in range(len(x)):
            max_x[i]=data[data.keys()[i]]
        x_adv=max_x
        x_adv_path[0,1]=max_x

        return x_adv,x_adv_path
            
    def generate(self,X,y):
        X_adv_path=np.zeros((X.shape[0],2,X.shape[1]))
        X_size=X.shape[1]
        num=X.shape[0]
        X_adv=np.zeros(shape=(num,X_size))
        for i in tqdm(range(num)):
            OpenboxMethod.count=0
            x,x_adv_path=self._get_single_x(X[i])
            X_adv[i]=x
            X_adv_path[i]=x_adv_path
        y_adv=y
        return X_adv,y_adv,X_adv_path
