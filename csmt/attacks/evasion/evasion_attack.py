'''
Author: your name
Date: 2021-04-01 17:38:22
LastEditTime: 2021-08-02 08:54:11
LastEditors: your name
Description: In User Settings Edit
FilePath: /Cyber-Security-ML-Toolbox/csmt/attacks/evasion/evasion_attack.py
'''
import sys
import math
import numpy as np
from csmt.attacks.evasion.hop_skip_jump import HopSkipJump
from csmt.attacks.evasion.boundary import BoundaryAttack
from csmt.attacks.evasion.wgan import WGANEvasionAttack
from csmt.attacks.evasion.gradient import BIMAttack, GradientEvasionAttack,FGSMAttack,PGDAttack,CWAttack,JSMAAttack,DeepFoolAttack,UniversalAttack
from csmt.attacks.evasion.de import DEEvasionAttack
from csmt.get_model_data import models_predict
from csmt.zoopt.DE import DE
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd
import copy
from csmt.zoopt.bayes_opt.bayesian_optimization import BayesianOptimization
from csmt.attacks.evasion.zoo_zhs import ZooAttack
from csmt.attacks.evasion.zones import ZONESMethod
from csmt.attacks.evasion.zosgd import ZOSGDMethod
from csmt.attacks.evasion.bayes_opt import BayesOptMethod
from csmt.attacks.evasion.grad_free_opt import GradFreeMethod
from csmt.attacks.evasion.openbox_opt import OpenboxMethod 

from csmt.attacks.evasion.zoscd import ZOSCDMethod
from csmt.attacks.evasion.zo_shap_scd import ZOShapSCDMethod
from csmt.attacks.evasion.zo_shap_sgd import ZOShapSGDMethod
from csmt.attacks.evasion.zoadamm import ZOAdaMMMethod
from csmt.attacks.evasion.zosgd_sum import ZOSGDSumMethod
from csmt.attacks.evasion.zosgd_shap_sum import ZOSGDShapSumMethod
from csmt.attacks.evasion.zoadamm_sum import ZOAdammSumMethod
from csmt.attacks.evasion.zones_adamm_sum import ZONESAdammSumMethod
from csmt.attacks.evasion.mimicry import MimicryMethod

from csmt.classifiers.scores import get_class_scores

def evasion_dict(model,algorithm,upper,lower,feature_importance,mask=None):

    if algorithm == 'zoo':
        attack = ZooAttack(classifier=model.classifier, learning_rate=0.01, max_iter=30,abort_early=True,nb_parallel=10, batch_size=1, variable_h=0.1)
        return attack
    if algorithm=='hsj':
        print(mask)
        attack=HopSkipJump(classifier=model.classifier, targeted=False, max_iter=5, max_eval=1000, init_eval=10, verbose=False)
        return attack
    if algorithm=='bound':
        attack=BoundaryAttack(estimator=model.classifier, targeted=False, max_iter=0, delta=0.001, epsilon=0.01,verbose=False)
        return attack
    evasion_dic={
        'gradient':GradientEvasionAttack,
        'fgsm':FGSMAttack,
        'fgsm_l1':FGSMAttack,
        'fgsm_l2':FGSMAttack,
        'universal':UniversalAttack,
        'pgd':PGDAttack,
        'pgd_l1':PGDAttack,
        'pgd_l2':PGDAttack,
        'cw':CWAttack,
        'bim':BIMAttack,
        'jsma':JSMAAttack,
        'deepfool':DeepFoolAttack,
        'zones':ZONESMethod,
        'zosgd':ZOSGDMethod,
        'zoscd':ZOSCDMethod,
        'bayes':BayesOptMethod,
        'openbox_opt':OpenboxMethod,
        'grad_free':GradFreeMethod,
        'de':DEEvasionAttack,
        'zoadamm':ZOAdaMMMethod,
        'zo_shap_sgd':ZOShapSGDMethod,
        'zo_shap_scd':ZOShapSCDMethod,
        'zosgd_sum':ZOSGDSumMethod
    }

    if 'l1' in algorithm:
        return evasion_dic[algorithm](estimator=model,eps=1.5,eps_step=1.5,max_iter=1,norm=1,upper=upper,lower=lower,feature_importance=feature_importance,mask=mask)
    elif 'l2' in algorithm:
        return evasion_dic[algorithm](estimator=model,eps=0.6,eps_step=0.6,max_iter=1,norm=2,upper=upper,lower=lower,feature_importance=feature_importance,mask=mask)
    else:
        return evasion_dic[algorithm](estimator=model,eps=0.1,eps_step=0.1,max_iter=1,norm=np.inf,upper=upper,lower=lower,feature_importance=feature_importance,mask=mask)



def EvasionAttack(attack_models,attack_model_name,trained_models,evasion_algorithm,X_test,y_test,upper=1,lower=0,feature_importance=None,mask=None):
    if len(attack_models)==1 and len(evasion_algorithm)==1:
        attack=evasion_dict(attack_models[0],evasion_algorithm[0],upper,lower,feature_importance,mask=mask)
        if evasion_algorithm[0]=='hsj':
            X_adv,y_adv,X_adv_path=attack.generate(X_test,y_test,mask=mask)
        else:
            X_adv,y_adv,X_adv_path=attack.generate(X_test,y_test)
    if len(attack_models)!=1 and len(evasion_algorithm)==1:
        X_adv,y_adv,X_adv_path=TransferEnsembleEvasionAttack(attack_models,attack_model_name,evasion_algorithm,X_test,y_test,upper=1,lower=0,feature_importance=None,mask=None)
        # X_adv,y_adv,X_adv_path=TransferBayesEnsembleEvasionAttack(attack_models,attack_model_name,trained_models,evasion_algorithm,X_test,y_test,upper=1,lower=0,feature_importance=None,mask=None)
    if len(attack_models)==1 and len(evasion_algorithm)!=1:
        X_adv,y_adv,X_adv_path=EnsembleEvasionAttack(attack_models,evasion_algorithm,X_test,y_test,upper=1,lower=0,feature_importance=None,mask=None)
        # X_adv,y_adv,X_adv_path=BayesEnsembleEvasionAttack(attack_models,evasion_algorithm,X_test,y_test,upper=1,lower=0,feature_importance=None,mask=None)
    if len(attack_models)!=1 and len(evasion_algorithm)!=1:
        print('迁移攻击不能使用多种攻击方式的集成!')
        sys.exit()
    return X_adv,y_adv,X_adv_path

def TransferEnsembleEvasionAttack(attack_models,attack_model_name,evasion_algorithm,X_test,y_test,upper=1,lower=0,feature_importance=None,mask=None):
    print("开始迁移集成攻击")
    print(attack_model_name)
    transfer_weight=1.0/len(attack_model_name)*np.ones(len(attack_model_name),dtype=float)
    X_adv_all=np.zeros((X_test.shape[0],X_test.shape[1]))
    for i in range(len(attack_model_name)):
        attack=evasion_dict(attack_models[i],evasion_algorithm[0],upper,lower,feature_importance)
        X_adv,y_adv,X_adv_path=attack.generate(X_test,y_test)
        X_adv_all+=X_adv*transfer_weight[i]
    return X_adv_all,y_adv,X_adv_path

def TransferBayesEnsembleEvasionAttack(attack_model,attack_model_name,evaluation_models,evasion_algorithm,X_test,y_test,upper=1,lower=0,feature_importance=None,mask=None):
    print("开始贝叶斯迁移集成攻击")
    print(attack_model_name)
    transfer_weight=1.0/len(attack_model_name)*np.ones(len(attack_model_name),dtype=float)
    
    def get_result(w):
        X_adv_all=np.zeros((X_test.shape[0],X_test.shape[1]))
        y_pred_all=np.zeros((X_test.shape[0],2))
        w_new=get_distribute(w,len(attack_model_name))
        for i in range(len(attack_model_name)):
            attack=evasion_dict(attack_model[i],evasion_algorithm[0],upper,lower,feature_importance)
            X_adv,y_adv,X_adv_path=attack.generate(X_test,y_test)
            X_adv_all+=X_adv*w_new[i]

        y_test_adv,y_pred_all=models_predict(evaluation_models,X_adv_all,y_test)
        
        #使得第一个评估模型的攻击成功率
        # y_pred_adv=np.argmax(y_pred_all[0], axis=1)
        # result=get_class_scores(y_test_adv, y_pred_adv)
        # #增加惩罚项
        # lamda=0
        # goal=(1-result[3])-lamda*np.square((np.sum(w)-1))

        #使得平均攻击成功率
        # K=len(evaluation_models)
        # score=0
        # for k in range(K):
        #     y_pred_adv=np.argmax(y_pred_all[k], axis=1)
        #     result=get_class_scores(y_test_adv, y_pred_adv)
        #     score+=(1-result[3])
        # goal=score/K

        #扩展到全部攻击成功率
        y_test_1=y_test
        y_pred_arr_1=y_pred_all

        K=len(evaluation_models)
        adv_maps = np.full((K,len(y_test_1)), False)
        for k in range(K):
            y_pred=np.argmax(y_pred_arr_1[k], axis=1)
            adv_maps[k]=(y_pred != y_test_1)
        asr_all = np.full(len(y_test_1), True)
        for adv_map in adv_maps:
            asr_all = np.logical_and(adv_map, asr_all)
        # print ('zhs_ASR_all: %.2f %%' % (100 * np.sum(asr_all) / float(len(y_test_1))))
        goal=(np.sum(asr_all) / float(len(y_test_1)))

        return goal

    bound=[]
    keys=[]
    for i in range(len(attack_model_name)):
        bound.append([0.01,0.99])
        keys.append('x'+str(i))

    bo = BayesianOptimization(f=get_result,pbounds={'x':bound},random_state=7)
    
    bo.maximize(init_points=10,n_iter=20,distribute=None)
    print(bo.max['params'])
    max_x=np.array([bo.max['params'][key] for key in keys])
    weight_distribute=get_distribute(max_x,len(attack_model_name))
    print(weight_distribute)  
    transfer_weight=weight_distribute

    X_adv_all=np.zeros((X_test.shape[0],X_test.shape[1]))
    for i in range(len(attack_model_name)):
        attack=evasion_dict(attack_model[i],evasion_algorithm[0],upper,lower,feature_importance)
        X_adv,y_adv,X_adv_path=attack.generate(X_test,y_test)
        X_adv_all+=X_adv*transfer_weight[i]
    return X_adv_all,y_adv,X_adv_path

def EnsembleEvasionAttack(attack_models,evasion_algorithm,X_test,y_test,upper=1,lower=0,feature_importance=None,mask=None):
    print("开始多种扰动集成攻击")
    evasion_algorithm_arr=evasion_algorithm
    evasion_weight=1.0/len(evasion_algorithm_arr)*np.ones(len(evasion_algorithm_arr),dtype=float)
    evasion_weight=get_distribute(evasion_weight,len(evasion_weight))
    X_adv_all=np.zeros((X_test.shape[0],X_test.shape[1]))
    for i in range(len(evasion_algorithm_arr)):
        attack=evasion_dict(attack_models[0],evasion_algorithm_arr[i],upper,lower,feature_importance)
        X_adv,y_adv,X_adv_path=attack.generate(X_test,y_test)
        X_adv_all+=X_adv*evasion_weight[i]
    return X_adv_all,y_adv,X_adv_path

def BayesEnsembleEvasionAttack(attack_models,evasion_algorithm,X_test,y_test,upper=1,lower=0,feature_importance=None,mask=None):
    print("开始多种扰动贝叶斯自动集成攻击")
    print(evasion_algorithm)
    evasion_algorithm_arr=evasion_algorithm
    evasion_weight=1.0/len(evasion_algorithm_arr)*np.ones(len(evasion_algorithm_arr),dtype=float)
    def get_result(w):
        X_adv_all=np.zeros((X_test.shape[0],X_test.shape[1]))
        y_pred_all=np.zeros((X_test.shape[0],2))
        w_new=get_distribute(w,len(evasion_algorithm_arr))

        for i in range(len(evasion_algorithm_arr)):
            attack=evasion_dict(attack_models[0],evasion_algorithm_arr[i],upper,lower,feature_importance)
            X_adv,y_adv,X_adv_path=attack.generate(X_test,y_test)
            X_adv_all+=X_adv*w_new[i]
        y_test_adv,y_pred_all=models_predict(attack_models,X_adv_all,y_test)
        y_pred_adv=np.argmax(y_pred_all[0], axis=1)
        result=get_class_scores(y_test_adv, y_pred_adv)
        #增加惩罚项
        lamda=0
        goal=(1-result[3])-lamda*np.square((np.sum(w)-1))
        p_w=[]
        for i in range(len(w_new)):
            p_w.append(round(w_new[i], 2))
        p_w.append(round(goal, 2))
        print(np.array(p_w))
        return goal
    bound=[]
    keys=[]
    for i in range(len(evasion_algorithm_arr)):
        bound.append([0.01,0.99])
        keys.append('x'+str(i))

    bo = BayesianOptimization(f=get_result,pbounds={'x':bound},random_state=7)
    
    bo.maximize(init_points=10,n_iter=20,distribute=None)
    print(bo.max['params'])
    max_x=np.array([bo.max['params'][key] for key in keys])
    weight_distribute=get_distribute(max_x,len(evasion_algorithm_arr))
    # print(weight_distribute)  
    evasion_weight=weight_distribute

    X_adv_all=np.zeros((X_test.shape[0],X_test.shape[1]))
    for i in range(len(evasion_algorithm_arr)):
        attack=evasion_dict(attack_models[0],evasion_algorithm_arr[i],upper,lower,feature_importance)
        X_adv,y_adv,X_adv_path=attack.generate(X_test,y_test)
        X_adv_all+=X_adv*evasion_weight[i]
    return X_adv_all,y_adv,X_adv_path



def get_distribute(max_x,len_distribute):
    x_all=0
    for i in range(len_distribute):
        x_all=x_all+max_x[i]
    distribute=[]
    for i in range(len_distribute):
        # distribute.append(format(max_x[i]/x_all, '.2f'))
        distribute.append(max_x[i]/x_all)
    return distribute




# def EnsembleEvasionAttack(attack_model,X_test,y_test,upper=1,lower=0,feature_importance=None,mask=None):

#     evasion_algorithm_arr=['fgsm','pgd']
#     evasion_weight=[0.5,0.5]
#     attack=evasion_dict(attack_model,evasion_algorithm_arr[0],upper,lower,feature_importance)
#     X_adv,y_adv,X_adv_path=attack.generate(X_test,y_test)
#     X_adv_path_all=np.zeros((X_adv_path.shape[0],X_adv_path.shape[1],X_adv_path.shape[2]))
#     X_adv_all=np.zeros((X_test.shape[0],X_test.shape[1]))
#     for i in range(len(evasion_algorithm_arr)):
#         attack=evasion_dict(attack_model,evasion_algorithm_arr[i],upper,lower,feature_importance)
#         X_adv,y_adv,X_adv_path=attack.generate(X_test,y_test)
#         X_adv_all+=X_adv*evasion_weight[i]
#         X_adv_path_all+=X_adv_path*evasion_weight[i]
#     return X_adv_all,y_adv,X_adv_path_all
    


    


    
    

