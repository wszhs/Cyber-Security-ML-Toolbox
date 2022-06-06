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
from csmt.defences.trainer.adversarial_trainer import AdversarialTrainer
    
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

def plot_func(n_classes,models,models_name,table):
    n_classes=np.unique(y_train).size
    fig = CFigure(width=5 * len(models), height=5 * 2)
    for i in range(len(trained_models)):
        fig.subplot(2, int(ceil(len(models) / 2)), i + 1)
        fig.sp.plot_ds(X_test,y_test)
        fig.sp.plot_decision_regions(models[i], n_grid_points=100,n_classes=n_classes)
        fig.sp.plot_fun(models[i].predict_abnormal, plot_levels=False, 
                        multipoint=True, n_grid_points=50,alpha=0.6)
        fig.sp.title(models_name[i])
        fig.sp.text(0.01, 0.01, "Accuracy on test set: {:.2%}".format(table['accuracy'].tolist()[i]))
    fig.show()

def plot_fun_path(n_classes,trained_models,orig_models_name,table,adv_path):
    fig = CFigure(width=5 * len(trained_models), height=5 * 2)

    for i in range(len(trained_models)):
        fig.subplot(2, int(ceil(len(trained_models) / 2)), i + 1)
        fig.sp.plot_ds(X_test,y_test)
        fig.sp.plot_decision_regions(trained_models[i], n_grid_points=100,n_classes=n_classes)
        for j in range(0,10):
            adv_path=X_adv_path[j,:]
            fig.sp.plot_path(adv_path)
        fig.sp.plot_fun(trained_models[i].predict_abnormal, plot_levels=False, 
                        multipoint=True, n_grid_points=50,alpha=0.6)
        fig.sp.title(orig_models_name[i])
        fig.sp.text(0.01, 0.01, "Accuracy on test set: {:.2%}".format(table['accuracy'].tolist()[i]))
    fig.show()


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
    # 绘制决策边界
    # plot_func(np.unique(y_train).size,trained_models,orig_models_name,table)

    X_adv,y_adv,X_adv_path=EvasionAttack(attack_models,attack_model_name,trained_models,evasion_algorithm,X_test_1,y_test_1)
        
    X_test_adv = np.append(X_test_0, X_adv, axis=0)
    y_test_adv = np.append(y_test_0, y_adv, axis=0)

    y_test_adv,y_pred_adv=models_predict(trained_models,X_test_adv,y_test_adv)

    print_results(datasets_name,orig_models_name,y_test_adv,y_pred_adv,'adversarial_accuracy')
    plot_fun_path(np.unique(y_train).size,trained_models,orig_models_name,table,X_adv_path)

    # 针对多种扰动集成对抗训练
    adv_train_models=AdversarialTrainer(datasets_name,attack_models,attack_model_name,trained_models,orig_models_name,adv_train_algorithm,X_train,y_train,X_val,y_val)

    # 对抗训练之后对抗样本的精度
    y_test_adv_train,y_pred_adv_train=models_predict(adv_train_models,X_test_adv,y_test_adv)

    table=print_results(datasets_name,orig_models_name,y_test_adv_train,y_pred_adv_train,'adv_train_accuracy')
    # 绘制决策边界
    # plot_func(np.unique(y_train).size,adv_train_models,orig_models_name,table)
    plot_fun_path(np.unique(y_train).size,adv_train_models,orig_models_name,table,X_adv_path)


    # 对抗训练之后原始的精度
    y_test_org,y_pred_org=models_predict(adv_train_models,X_test,y_test)

    print_results(datasets_name,orig_models_name,y_test_org,y_pred_org,'adv_train_org_accuracy')

