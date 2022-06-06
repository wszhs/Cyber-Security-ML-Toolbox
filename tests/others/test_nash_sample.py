'''
Author: your name
Date: 2021-04-24 12:30:42
LastEditTime: 2021-04-25 16:35:38
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /Cyber-Security-ML-Toolbox/tests/test_nash_sample.py
'''
import sys
sys.path.append("/Users/zhanghangsheng/Documents/my_code/Cyber-Security-ML-Toolbox")
import numpy as np
from csmt.zoopt.bayes_opt.bayesian_optimization import BayesianOptimization

def get_distribute(max_x,len_distribute):
    x_all=0
    for i in range(len_distribute):
        x_all=x_all+max_x[i]
    distribute=[]
    for i in range(len_distribute):
        distribute.append(max_x[i]/x_all)
    return distribute

def Nash():

    np.random.seed(1)

    def calReward (action_1, action_2):
        reward_1 = 0
        reward_2 = 0
        if (action_1, action_2) == (0, 0):
            reward_1 = -6
            reward_2 = -6
        elif (action_1, action_2) == (0, 1):
            reward_1 = 0
            reward_2 = 9
        elif (action_1, action_2) == (1, 0):
            reward_1 = -9
            reward_2 = 0
        elif (action_1, action_2) == (1, 1):
            reward_1 = -1
            reward_2 = -1
        return (reward_1, reward_2)

    A_distribute=[0.2,0.8]
    B_distribute=[0.8,0.2]

    def get_score(A_distribute,B_distribute):
        reward_all=0
        for i in range(10000):
            action_A=np.random.choice(a=[0,1],p=A_distribute)
            action_B=np.random.choice(a=[0,1],p=B_distribute)
            reward_A,reward_B=calReward(action_A,action_B)
            reward_all=reward_all+reward_A
        return reward_all


    def get_A_score(X_distribute):
        A_distribute=get_distribute(X_distribute,2)
        reward_all=0
        for i in range(10000):
            action_A=np.random.choice(a=[0,1],p=A_distribute)
            action_B=np.random.choice(a=[0,1],p=B_distribute)
            reward_A,reward_B=calReward(action_A,action_B)
            reward_all=reward_all+reward_A
        return reward_all

    def get_B_score(X_distribute):
        # print(A_distribute)
        B_distribute=get_distribute(X_distribute,2)
        reward_all=0
        for i in range(1000):
            action_A=np.random.choice(a=[0,1],p=A_distribute)
            action_B=np.random.choice(a=[0,1],p=B_distribute)
            reward_A,reward_B=calReward(action_A,action_B)
            reward_all=reward_all+reward_B
        return reward_all

        # 实例化一个bayes优化对象
    bound=[]
    keys=[]
    for i in range(len(A_distribute)):
        bound.append([0.01,0.99])
        keys.append('X_distribute'+str(i))


    for i in range(10):

        bo_A = BayesianOptimization(
        get_A_score,
        {'X_distribute':bound}
        )
        
        bo_B = BayesianOptimization(
        get_B_score,
        {'X_distribute':bound}
        )

        bo_A.maximize(n_iter=1)
        max_A_x=np.array([bo_A.max['params'][key] for key in keys ])
        A_distribute=get_distribute(max_A_x,len(A_distribute))
        print('A')
        print(A_distribute)

        bo_B.maximize(n_iter=1)
        max_B_x=np.array([bo_B.max['params'][key] for key in keys ])
        B_distribute=get_distribute(max_B_x,len(B_distribute))
        print('B')
        print(B_distribute)

if __name__=='__main__':
    Nash()

    

    
            
            

        

