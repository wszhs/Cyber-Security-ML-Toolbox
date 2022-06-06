'''
Author: your name
Date: 2021-04-16 17:30:14
LastEditTime: 2021-04-30 19:05:12
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /Cyber-Security-ML-Toolbox/tests/test_attack.py
'''
'''
Author: your name
Date: 2021-04-01 16:39:21
LastEditTime: 2021-04-16 15:16:03
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /Cyber-Security-ML-Toolbox/tests/test.py
'''
import sys
sys.path.append("/Users/zhanghangsheng/Documents/my_code/Cyber-Security-ML-Toolbox")
from csmt.get_model_data import get_datasets,parse_arguments,train_models,load_models,print_results,print_results_single
from csmt.attacks.evasion.evasion_attack import EvasionAttack,EnsembleEvasionAttack,AutoEnsembleEvasionAttack,BayesEnsembleEvasionAttack
from csmt.defences.trainer.adversarial_trainer import AdversarialTrainer,EnsembleAdversarialTrainer,BayesEnsembleAdversarialTrainer
from csmt.ensemble.ensemble_classifier import EnsembleClassifier
import numpy as np

if __name__=='__main__':

    
    arguments = sys.argv[1:]
    options = parse_arguments(arguments)

    
    X_train,y_train,X_test,y_test,n_features=get_datasets(options)
    
    # for i in range(0,len(X_train)):
    #     trained_models=train_models(options,False,X_train[i],y_train[i])
    #     print('zhs')
    #     print_results_single(options,trained_models,X_test[i],y_test[i],str(i),'DeepFool','original_accuracy')
    #     X_test[i]=X_test[i][y_test[i]==1]
    #     y_test[i]=y_test[i][y_test[i]==1]
    #     X_adv,y_adv=EvasionAttack(options.datasets,'lr_gradient',X_test[i],y_test[i])
    #     print_results_single(options,trained_models,X_adv,y_adv,str(i),'DeepFool','adversarial_accuracy')


    trained_models=train_models(options,False,X_train,y_train)
    # trained_models=load_models(options,n_features)
    # # # #print result before adversarial attack
    print_results(options,trained_models,X_test,y_test,'original_accuracy')

    # EnsembleClassifier(options,X_train,y_train,X_test,y_test,trained_models)
    X_test=X_test[y_test==1]
    y_test=y_test[y_test==1]
    # EnsembleClassifier(options,X_train,y_train,X_test,y_test,trained_models)
    # # # start adversarial attack
    X_adv,y_adv=EvasionAttack(options.datasets,'lr_gradient',X_test,y_test)
    print_results(options,trained_models,X_adv,y_adv,'adversarial_accuracy')


    # for i in range(0,len(X_adv)):
    #     # dis=np.sqrt(np.sum((X_adv[i] - X_test[i])**2))
    #     # dis=np.linalg.norm(X_adv[i] - X_test[i],ord=np.inf)
    #     dis=np.linalg.norm(X_adv[i] - X_test[i],ord=2)
    #     print(dis)

    

    