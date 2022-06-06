
import copy
import numpy as np
from csmt.attacks.evasion.util import get_distribute

class AbstractEvasionSum:
    def __init__(self,estimator,eps,eps_step,max_iter,norm,upper,lower,feature_importance):
        self.estimator=estimator
        self.norm=norm
        self.eps=eps
        self.eps_step=eps_step
        self.max_iter=max_iter
        self.upper=upper
        self.lower=lower
        self.feature_importance=feature_importance
        self.q = 20 ### number of random direction vectors
        self.mu = 0.05 ### key parameter: smoothing parameter in ZO gradient estimation # 0.001 for imagenet
        self.kappa = 1e-10
        self.lmd = 1
        self.beta= 0.01
        self.init_const = 1 ### regularization parameter prior to attack loss

    def generate(self, x, y):
        X_adv=copy.deepcopy(x)
        X_adv_path=np.zeros((X_adv.shape[0],self.max_iter+1,X_adv.shape[1]))
        for i in range(x.shape[0]):
            if i%1==0:
                print(i)
            x_singe=x[i:i+1]
            y_single=y[i:i+1]
            x_adv_single,x_adv_path=self.generate_single(x_singe,y_single)
            X_adv[i]=x_adv_single
            X_adv_path[i]=x_adv_path
        y_adv=y
        return X_adv,y_adv,X_adv_path

    def generate_single(self,x,y):
        # print('next')
        const=self.init_const
        x_orig=copy.deepcopy(x)
        x_adv=copy.deepcopy(x)
        delta_adv = np.zeros((1,self.max_iter,x.shape[1]))
        len_models=len(self.estimator.models_name)
        weights=np.ones(len_models,dtype=np.float32)*1.0/len_models
        x_adv_path=np.zeros((1,self.max_iter+1,x.shape[1]))
        total_loss = np.zeros((self.max_iter,len_models))
        iter=0
        for i in range(0,self.max_iter):
            # base_lr = self.eps_step/np.sqrt(i+1)
            base_lr = self.eps_step
            grad_est=self.gradient_estimation_sum(self.mu,self.q,x,self.kappa,y,const,weights)
            if self.norm in [np.inf, "inf"]:
                delta_adv[0,i] =delta_adv[0,i-1]-base_lr*np.sign(grad_est)
                delta_adv[0,i]=np.clip(delta_adv[0,i],-self.eps,self.eps)
                delta_adv[0,i] = np.clip(x_adv+delta_adv[0,i], self.lower, self.upper) - x_adv
            x_adv_path[0,i+1]=x+delta_adv[0,i]
            x_adv_tmp=x_orig+delta_adv[0,i]

            total_loss[i]=self.function_evaluation_cons_models(x_orig+delta_adv[0,i],y,weights)
            w_grad = total_loss[i] - 2 * self.lmd * (weights-1/(len_models))
            w_proj = weights + self.beta* w_grad
            weights = get_distribute(w_proj)
            print(weights)
            
            finish_count=0
            for k in range(len_models):
                if self.estimator.predict(x_adv_tmp)[k,0,y]<self.estimator.predict(x_adv_tmp)[k,0,0]:
                    finish_count+=1
            if finish_count==len_models:
                iter=i
                break
        # iter=self.max_iter-1   
        x_adv = x_adv+delta_adv[0,iter]
        return x_adv,x_adv_path

    def function_evaluation_cons_models(self,x_adv,target_label,weights):
        len_models=len(self.estimator.models_name)
        orig_prob = self.estimator.predict(x_adv)
        Loss=np.zeros(len_models)
        for i in range(len_models):
            Loss[i]=weights[i]*orig_prob[i][0, target_label] 
        return Loss

    def function_evaluation_cons_sum(self,x_adv, kappa, target_label, const,x,weights):
        orig_prob = self.estimator.predict(x_adv)
        Loss1=0
        len_models=len(self.estimator.models_name)
        for i in range(len_models):
            Loss1+=weights[i]*orig_prob[i][0, target_label] 
        Loss2 = np.linalg.norm(x_adv - x) ** 2 ### squared normx
        return Loss1, Loss2

    def gradient_estimation_sum(self,mu,q,x,kappa,target_label,const,weights):
        pass

