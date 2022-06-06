'''
Author: your name
Date: 2021-06-07 14:44:16
LastEditTime: 2021-06-07 14:50:24
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /Cyber-Security-ML-Toolbox/tests/evasion_attack/test_standard_attack.py
'''
'''
Author: your name
Date: 2021-04-01 15:20:27
LastEditTime: 2021-06-07 12:54:59
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /Cyber-Security-ML-Toolbox/tests/test_data.py
'''
import sys
ROOT_PATH='/Users/zhanghangsheng/Documents/my_code/Cyber-Security-ML-Toolbox'
sys.path.append(ROOT_PATH)
import numpy as np
from csmt.get_model_data import get_datasets, models_train,parse_arguments,models_train,models_load,print_results,models_and_ensemble_predict,models_predict
from csmt.attacks.evasion.evasion_attack import EvasionAttack,EnsembleEvasionAttack,BayesEnsembleEvasionAttack
from csmt.ensemble.ensemble_classifier import EnsembleClassifier

if __name__=='__main__':
    arguments = sys.argv[1:]
    options = parse_arguments(arguments)
    datasets_name=options.datasets
    models_name=options.algorithms
    
    X_train,y_train,X_val,y_val,X_test,y_test,n_features,mask=get_datasets(options)
    # print(mask)
    
    trained_models=models_train(options,False,X_train,y_train)
    y_test,y_pred_arr=models_predict(trained_models,X_test,y_test)
    # # #print result before adversarial attack
    print_results(datasets_name,models_name,y_test,y_pred_arr,'original_accuracy')

    
    X_adv,y_adv=EvasionAttack(datasets_name,'lr_gradient',X_test,y_test,mask)
    y_test_adv,y_test_pred_arr=models_predict(trained_models,X_adv,y_adv)
    print_results(datasets_name,models_name,y_test_adv,y_test_pred_arr,'adversarial_accuracy')
