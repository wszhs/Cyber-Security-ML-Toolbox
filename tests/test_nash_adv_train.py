import sys
ROOT_PATH='/Users/zhanghangsheng/Documents/my_code/Cyber-Security-ML-Toolbox'
sys.path.append(ROOT_PATH)
import numpy as np
from csmt.get_model_data import get_datasets,models_train,parse_arguments,models_train,print_results,models_predict
from csmt.attacks.evasion.evasion_attack import EvasionAttack
from csmt.Interpretability.shap.get_interaction import get_average_pairwise_interaction
import torch
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
#torch.set_default_tensor_type(torch.DoubleTensor)
import random
from csmt.figure import CFigure
from math import ceil
import shap_zhs as shap
from csmt.defences.trainer.adversarial_trainer import AdversarialTrainer,NashAdversarialTrainer,WeightAdversarialTrainer
    
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
# 设置随机数种子
setup_seed(20)

def plot_headmap(X,a_score,model_name):
    X_x=X[:,0]
    X_y=X[:,1]
    plt.scatter(X_x, X_y, marker='o', c=a_score, cmap='viridis')
    plt.colorbar()
    plt.xticks(size=14)
    plt.yticks(size=14)
    plt.title(model_name)
    plt.show()

if __name__=='__main__':
    arguments = sys.argv[1:]
    options = parse_arguments(arguments)
    datasets_name=options.datasets
    attack_model_name=options.attack_models
    orig_models_name=options.algorithms
    evasion_algorithm=options.evasion_algorithm
    #针对多种扰动方法集成的对抗训练
    adv_train_algorithm=options.adv_train_algorithm
    X_train,y_train,X_val,y_val,X_test,y_test,mask=get_datasets(options)
 

    setup_seed(20)
    attack_models=models_train(datasets_name,attack_model_name,False,X_train,y_train,X_val,y_val)
    setup_seed(20)
    trained_models=models_train(datasets_name,orig_models_name,False,X_train,y_train,X_val,y_val)
    y_test,y_pred=models_predict(trained_models,X_test,y_test)

    X_test_1=X_test[y_test==1]
    y_test_1=y_test[y_test==1]
    X_test_0=X_test[y_test==0]
    y_test_0=y_test[y_test==0]
    print(X_test_1.shape)
    print(X_test_0.shape)

    table=print_results(datasets_name,orig_models_name,y_test,y_pred,'original_accuracy')

    # 针对多种扰动集成对抗训练
    NashAdversarialTrainer(datasets_name,attack_models,attack_model_name,trained_models,orig_models_name,evasion_algorithm,adv_train_algorithm,X_train,y_train,X_test,y_test,X_val,y_val)





