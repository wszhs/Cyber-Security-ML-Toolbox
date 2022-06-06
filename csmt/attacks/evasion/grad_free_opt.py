
import numpy as np

from csmt.zoopt.gradient_free_optimizers  import RandomSearchOptimizer,BayesianOptimizer,HillClimbingOptimizer,StochasticHillClimbingOptimizer,SimulatedAnnealingOptimizer,GridSearchOptimizer,ParticleSwarmOptimizer,EvolutionStrategyOptimizer,OneDimensionalBayesianOptimization
from csmt.zoopt.gradient_free_optimizers  import BayesianOptimizer
from tqdm import tqdm
np.random.seed(10)
import time

class GradFreeMethod():
    estimator=None
    
    count=0

    def get_score(x):

        GradFreeMethod.count=GradFreeMethod.count+1
        # print(GradFreeMethod.count)

        p=np.zeros(len(x))
        for i in range(len(x)):
            p[i]=x['x'+str(i)]
        score=GradFreeMethod.estimator.predict(p.reshape(1,-1))
        return score[0][0]

    def __init__(self,estimator,eps,eps_step,max_iter,norm,upper,lower,feature_importance,mask):
        GradFreeMethod.estimator=estimator
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

        search_space={}
        for i in range(x.shape[0]):
            bound=np.array([-self.eps,self.eps])+x[i]
            bound=np.clip(bound, 0, 1)
            search_space.update({'x'+str(i):np.arange(bound[0],bound[1],0.001)})

        # opt = HillClimbingOptimizer(search_space,random_state=20)
        # opt = RandomSearchOptimizer(search_space,random_state=20)
        # opt = BayesianOptimizer(search_space,random_state=20)
        # opt=TreeStructuredParzenEstimators(search_space)
        # opt=HillClimbingOptimizer(search_space)
        # opt=StochasticHillClimbingOptimizer(search_space)
        opt=GridSearchOptimizer(search_space,random_state=20)
        # opt=ParticleSwarmOptimizer(search_space,random_state=20)
        # opt=EvolutionStrategyOptimizer(search_space,random_state=20)
        # opt=OneDimensionalBayesianOptimization(search_space)
        opt.search(GradFreeMethod.get_score, n_iter=100,verbosity=False)

        history=opt.score_l
        history=np.array(history)
        # print(history)
        # print(opt.best_score)

        max_x=np.zeros(len(x))
        for i in range(len(x)):
            max_x[i]=opt.best_para['x'+str(i)]
        x_adv=max_x
        x_adv_path[0,1]=max_x

        return x_adv,x_adv_path
            
    def generate(self,X,y):
        X_adv_path=np.zeros((X.shape[0],2,X.shape[1]))
        X_size=X.shape[1]
        num=X.shape[0]
        X_adv=np.zeros(shape=(num,X_size))
        for i in tqdm(range(num)):
            GradFreeMethod.count=0
            x,x_adv_path=self._get_single_x(X[i])
            X_adv[i]=x
            X_adv_path[i]=x_adv_path
        y_adv=y
        return X_adv,y_adv,X_adv_path
