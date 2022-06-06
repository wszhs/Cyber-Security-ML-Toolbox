'''
Author: your name
Date: 2021-04-21 16:13:31
LastEditTime: 2021-05-19 21:35:25
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /Cyber-Security-ML-Toolbox/tests/test_cross_trans.py
'''
import sys
sys.path.append("/Users/zhanghangsheng/Documents/my_code/Cyber-Security-ML-Toolbox")
# from csmt.get_model_data import get_datasets,parse_arguments,models_train,print_results_cross_trans,models_and_ensemble_predict
# from csmt.attacks.evasion.evasion_attack import EvasionAttack
import numpy as np
import pandas as pd
from plot import plot_transfer

if __name__=='__main__':

    
    # arguments = sys.argv[1:]
    # options = parse_arguments(arguments)
    # models_name=options.algorithms
    # datasets_name=options.datasets
    
    # X_train,y_train,X_val,y_val,X_test,y_test,n_features,mask=get_datasets(options)

    # trained_models=models_train(options,False,X_train,y_train)

    # X_test_1=X_test[y_test==1]
    # y_test_1=y_test[y_test==1]
    # X_test_0=X_test[y_test==0]
    # y_test_0=y_test[y_test==0]

    # adv_arr=['lr_fgsm','lr_pgd','svm_fgsm','svm_pgd','mlp_fgsm','mlp_pgd','tree']
    # X_adv_arr=[]
    # y_adv_arr=[]
    # for i in range(0,len(adv_arr)):
    #     X_adv,y_adv=EvasionAttack(datasets_name,adv_arr[i],X_test_1,y_test_1,mask)
    #     X_test_adv = np.append(X_test_0, X_adv, axis=0)
    #     y_test_adv = np.append(y_test_0, y_adv, axis=0)
    #     X_adv_arr.append(X_test_adv)
    #     y_adv_arr.append(y_test_adv)

    # table=print_results_cross_trans(models_name,adv_arr,trained_models,X_adv_arr,y_adv_arr)
    # print(table)
    table = pd.read_csv('experiments/plot/dohbrw/'+'cross_trans.csv', encoding='utf8', low_memory=False)
    # print(table)
    # table.to_csv('experiments/plot/'+datasets_name+'/'+'cross_trans.csv',index=False)
    plot_transfer(table,"dohbrw")
    

