'''
Author: your name
Date: 2021-04-06 09:34:48
LastEditTime: 2021-05-19 23:00:10
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /Cyber-Security-ML-Toolbox/csmt/defences/trainer/adversarial_trainer.py
'''
import numpy as np
import sys
from csmt.get_model_data import get_datasets,models_train,parse_arguments,models_train,print_results,models_predict
from csmt.get_model_data import models_train
from csmt.attacks.evasion.evasion_attack import EvasionAttack,evasion_dict
from csmt.zoopt.bayes_opt.bayesian_optimization import BayesianOptimization
from csmt.classifiers.scores import get_class_scores

def get_distribute(max_x,len_distribute):
    x_all=0
    for i in range(len_distribute):
        x_all=x_all+max_x[i]
    distribute=[]
    for i in range(len_distribute):
        distribute.append(max_x[i]/x_all)
    return distribute

def get_DSR_ALL(models_name,y_test,y_pred_arr):
    #提取黑样本
    y_test_1=y_test[y_test==1]
    y_pred_arr_1=y_pred_arr[:,y_test==1]

    K=len(models_name)
    adv_maps = np.full((K,len(y_test_1)), False)
    for k in range(K):
        y_pred=np.argmax(y_pred_arr_1[k], axis=1)
        adv_maps[k]=(y_pred == y_test_1)
    dsr_all = np.full(len(y_test_1), True)
    for adv_map in adv_maps:
        dsr_all = np.logical_and(adv_map, dsr_all)
    return (100 * np.sum(dsr_all) / float(len(y_test_1)))

def AdversarialTrainer(datasets_name,attack_models,attack_model_name,trained_models,orig_models_name,adv_train_algorithm,X_train,y_train,X_val,y_val):
    X_train_1=X_train[y_train==1]
    y_train_1=y_train[y_train==1]
    evasion_weight=evasion_weight=1.0/len(adv_train_algorithm)*np.ones(len(adv_train_algorithm),dtype=float)
    X_adv,y_adv,X_adv_path=WeightEvasionAttack(evasion_weight,attack_models,adv_train_algorithm,X_train_1,y_train_1)
    X_train_adv = np.append(X_train, X_adv, axis=0)
    y_train_adv = np.append(y_train, y_adv, axis=0)
    return models_train(datasets_name,orig_models_name,True,X_train_adv,y_train_adv,X_val,y_val)

def WeightEvasionAttack(evasion_weight,attack_models,evasion_algorithm,X_test,y_test,upper=1,lower=0,feature_importance=None,mask=None):
    print("开始多种扰动集成攻击")
    evasion_algorithm_arr=evasion_algorithm
    X_adv_all=np.zeros((X_test.shape[0],X_test.shape[1]))
    for i in range(len(evasion_algorithm_arr)):
        attack=evasion_dict(attack_models[0],evasion_algorithm_arr[i],upper,lower,feature_importance)
        X_adv,y_adv,X_adv_path=attack.generate(X_test,y_test)
        X_adv_all+=X_adv*evasion_weight[i]
    return X_adv_all,y_adv,X_adv_path

def WeightAdversarialTrainer(datasets_name,attack_models,attack_model_name,trained_models,orig_models_name,evasion_algorithm,adv_train_algorithm,X_train,y_train,X_test,y_test,X_val,y_val):
    X_train_1=X_train[y_train==1]
    y_train_1=y_train[y_train==1]
    X_test_1=X_test[y_test==1]
    y_test_1=y_test[y_test==1]
    X_test_0=X_test[y_test==0]
    y_test_0=y_test[y_test==0]

    X_adv,y_adv,X_adv_path=EvasionAttack(attack_models,attack_model_name,trained_models,evasion_algorithm,X_test_1,y_test_1)
    X_test_adv = np.append(X_test_0, X_adv, axis=0)
    y_test_adv = np.append(y_test_0, y_adv, axis=0)
    y_test_adv,y_pred_adv=models_predict(trained_models,X_test_adv,y_test_adv)
    print_results(datasets_name,orig_models_name,y_test_adv,y_pred_adv,'adversarial_accuracy')

    weight_distribute=1.0/len(adv_train_algorithm)*np.ones(len(adv_train_algorithm),dtype=float)
    adversarial_train_weight=weight_distribute

    X_adv,y_adv,X_adv_path=WeightEvasionAttack(adversarial_train_weight,attack_models,adv_train_algorithm,X_train_1,y_train_1)
    X_train_adv = np.append(X_train, X_adv, axis=0)
    y_train_adv = np.append(y_train, y_adv, axis=0)
    adv_train_models=models_train(datasets_name,orig_models_name,True,X_train_adv,y_train_adv,X_val,y_val)

    y_test_adv_train,y_pred_adv_train=models_predict(adv_train_models, X_test_adv,y_test_adv)

    table=print_results(datasets_name,orig_models_name,y_test_adv_train,y_pred_adv_train,'adv_train_accuracy')
    
def NashAdversarialTrainer(datasets_name,attack_models,attack_model_name,trained_models,orig_models_name,evasion_algorithm,adv_train_algorithm,X_train,y_train,X_test,y_test,X_val,y_val):
    X_train_1=X_train[y_train==1]
    y_train_1=y_train[y_train==1]
    X_test_1=X_test[y_test==1]
    y_test_1=y_test[y_test==1]
    X_test_0=X_test[y_test==0]
    y_test_0=y_test[y_test==0]

    X_adv,y_adv,X_adv_path=EvasionAttack(attack_models,attack_model_name,trained_models,evasion_algorithm,X_test_1,y_test_1)
    X_test_adv = np.append(X_test_0, X_adv, axis=0)
    y_test_adv = np.append(y_test_0, y_adv, axis=0)
    y_test_adv,y_pred_adv=models_predict(trained_models,X_test_adv,y_test_adv)
    print_results(datasets_name,orig_models_name,y_test_adv,y_pred_adv,'adversarial_accuracy')

    #生成对抗样本组
    X_test_eva_arr=[]
    for i in range(len(evasion_algorithm)):
        X_adv,y_adv,X_adv_path=EvasionAttack(attack_models,attack_model_name,trained_models,[evasion_algorithm[i]],X_test_1,y_test_1)
        X_test_eva = np.append(X_test_0, X_adv, axis=0)
        X_test_eva_arr.append(X_test_eva)
        y_test_eva = np.append(y_test_0, y_adv, axis=0)
    
    def get_result(w):
 
        w_new=get_distribute(w,len(adv_train_algorithm))
        adversarial_train_weight=w_new
        X_adv,y_adv,X_adv_path=WeightEvasionAttack(adversarial_train_weight,attack_models,adv_train_algorithm,X_train_1,y_train_1)
        X_train_adv = np.append(X_train, X_adv, axis=0)
        y_train_adv = np.append(y_train, y_adv, axis=0)
        adv_train_models=models_train(datasets_name,orig_models_name,True,X_train_adv,y_train_adv,X_val,y_val)
        y_pred_adv_train_arr=np.zeros((len(evasion_algorithm),X_test_eva_arr[0].shape[0],2))
        for i in range(len(evasion_algorithm)):
            y_test_adv_train,y_pred_adv_train=models_predict(adv_train_models, X_test_eva_arr[i],y_test_eva)
            y_pred_adv_train_arr[i]=y_pred_adv_train

        goal=get_DSR_ALL(evasion_algorithm,y_test_adv_train,y_pred_adv_train_arr)
        
        p_w=[]
        for i in range(len(w_new)):
            p_w.append(round(w_new[i], 2))
        p_w.append(round(goal, 2))
        print(np.array(p_w))

        return goal
    bound=[]
    keys=[]
    for i in range(len(adv_train_algorithm)):
        bound.append([0.01,0.99])
        keys.append('x'+str(i))

    bo = BayesianOptimization(f=get_result,pbounds={'x':bound},random_state=7)
    
    bo.maximize(init_points=10,n_iter=20,distribute=None)
    print(bo.max['params'])
    max_x=np.array([bo.max['params'][key] for key in keys])
    weight_distribute=get_distribute(max_x,len(adv_train_algorithm))
    print(weight_distribute)  
    adversarial_train_weight=weight_distribute

    X_adv,y_adv,X_adv_path=WeightEvasionAttack(adversarial_train_weight,attack_models,adv_train_algorithm,X_train_1,y_train_1)
    X_train_adv = np.append(X_train, X_adv, axis=0)
    y_train_adv = np.append(y_train, y_adv, axis=0)
    adv_train_models=models_train(datasets_name,orig_models_name,True,X_train_adv,y_train_adv,X_val,y_val)

    y_test_adv_train,y_pred_adv_train=models_predict(adv_train_models, X_test_adv,y_test_adv)

    table=print_results(datasets_name,orig_models_name,y_test_adv_train,y_pred_adv_train,'adv_train_accuracy')
    
def TransferAdversarialTrainer(datasets_name,adv_orig_models,adv_orig_models_name,trained_models,orig_models_name,evasion_algorithm,X_train,y_train,X_val,y_val):
    X_train_1=X_train[y_train==1]
    y_train_1=y_train[y_train==1]
    X_adv,y_adv,X_adv_path=EvasionAttack(adv_orig_models,adv_orig_models_name,trained_models,evasion_algorithm,X_train_1,y_train_1)
    X_train = np.append(X_train, X_adv, axis=0)
    y_train = np.append(y_train, y_adv, axis=0)
    return models_train(datasets_name,orig_models_name,True,X_train,y_train,X_val,y_val)

def WeightTransferAdversarialTrainer(adv_weight,datasets_name,adv_orig_models,adv_orig_models_name,trained_models,orig_models_name,evasion_algorithm,X_train,y_train,X_val,y_val):
    X_train_1=X_train[y_train==1]
    y_train_1=y_train[y_train==1]
    X_adv,y_adv,X_adv_path=WeightTransferEnsembleEvasionAttack(adv_weight,adv_orig_models,adv_orig_models_name,evasion_algorithm,X_train_1,y_train_1,upper=1,lower=0,feature_importance=None,mask=None)
    X_train = np.append(X_train, X_adv, axis=0)
    y_train = np.append(y_train, y_adv, axis=0)
    return models_train(datasets_name,orig_models_name,True,X_train,y_train,X_val,y_val)

def WeightTransferEnsembleEvasionAttack(transfer_weight,attack_models,attack_model_name,evasion_algorithm,X_test,y_test,upper=1,lower=0,feature_importance=None,mask=None):
    print("开始迁移集成攻击")
    print(attack_model_name)
    X_adv_all=np.zeros((X_test.shape[0],X_test.shape[1]))
    for i in range(len(attack_model_name)):
        attack=evasion_dict(attack_models[i],evasion_algorithm[0],upper,lower,feature_importance)
        X_adv,y_adv,X_adv_path=attack.generate(X_test,y_test)
        X_adv_all+=X_adv*transfer_weight[i]
    return X_adv_all,y_adv,X_adv_path

def NashTransferAdversarialTrainer(datasets_name,attack_models,attack_model_name,adv_orig_models,adv_orig_models_name,trained_models,orig_models_name,evasion_algorithm,X_train,y_train,X_val,y_val,X_test,y_test):
    evasion_weight=[0.99,0.01]
    adv_weight=[0.01,0.99]
    X_test_1=X_test[y_test==1]
    y_test_1=y_test[y_test==1]
    X_test_0=X_test[y_test==0]
    y_test_0=y_test[y_test==0]

    X_adv,y_adv,X_adv_path=WeightTransferEnsembleEvasionAttack(evasion_weight,attack_models,attack_model_name,evasion_algorithm,X_test,y_test,upper=1,lower=0,feature_importance=None,mask=None)
    X_test_adv = np.append(X_test_0, X_adv, axis=0)
    y_test_adv = np.append(y_test_0, y_adv, axis=0)

    y_test_adv,y_pred_adv=models_predict(trained_models,X_test_adv,y_test_adv)

    print_results(datasets_name,orig_models_name,y_test_adv,y_pred_adv,'adversarial_accuracy')

    # 针对迁移集成的对抗训练
    adv_train_models=WeightTransferAdversarialTrainer(adv_weight,datasets_name,adv_orig_models,adv_orig_models_name,trained_models,orig_models_name,evasion_algorithm,X_train,y_train,X_val,y_val)

    # 对抗训练之后对抗样本的精度
    y_test_adv_train,y_pred_adv_train=models_predict(adv_train_models,X_test_adv,y_test_adv)

    print_results(datasets_name,orig_models_name,y_test_adv_train,y_pred_adv_train,'adv_train_accuracy')



    




    




