import sys
sys.path.append("/Users/zhanghangsheng/Documents/my_code/Cyber-Security-ML-Toolbox")
import matplotlib
import csmt.Interpretability.sage as sage

import matplotlib.pyplot as plt
import csmt.figure.visualml.visualml as vml
import pandas as pd
from csmt.get_model_data import get_datasets,parse_arguments,models_train,print_results,models_predict
import numpy as np
import torch
import seaborn as sns
import csmt.Interpretability.shap as shap
from csmt.feature_selection import selectGa
# from xgboost import plot_importance
from csmt.figure.visualml.plot_importance import plot_xg_importance,plot_feature_importance_all,plot_dot


def plot_heatmap(table):
    matplotlib.style.use('seaborn-whitegrid')
    # table = np.random.rand(10, 12)
    sns.heatmap(table,vmin=0,vmax=1, cmap='viridis',annot=True)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14,rotation=0, horizontalalignment= 'right')
    plt.show()

if __name__=='__main__':
    arguments = sys.argv[1:]
    options = parse_arguments(arguments)
    datasets_name=options.datasets
    orig_models_name=options.algorithms
    evasion_algorithm=options.evasion_algorithm

    X_train,y_train,X_val,y_val,X_test,y_test,mask=get_datasets(options)

    trained_models=models_train(datasets_name,orig_models_name,False,X_train,y_train,X_val,y_val)

    y_test,y_pred=models_predict(trained_models,X_test,y_test)
    print_results(datasets_name,orig_models_name,y_test,y_pred,'original_accuracy')

    # 核夏普利值
    # explainer=shap.KernelExplainer(trained_models[0].predict,X_test)
    # explainer=shap.Explainer(trained_models[0].classifier.model,X_test)

    # 树夏普利值
    # explainer = shap.TreeExplainer(trained_models[0].classifier.model)
    # explainer=shap.Explainer(trained_models[0].classifier.model,X_test)

    # 局部解释
    # shap_values = explainer.shap_values(X_test)
    # shap.summary_plot(shap_values,plot_type="bar")
    # 博弈交互值
    # shap_interaction_values=explainer.shap_interaction_values(X_test)
    # print(shap_interaction_values[0:10])
    # shap.summary_plot(shap_interaction_values, max_display=10)

    # interaction_arr=np.zeros((X_test.shape[1],X_test.shape[1]))
    # for i in range(X_test.shape[1]):
    #     for j in range(X_test.shape[1]):
    #         interaction_arr[i,j]=round(np.abs(shap_interaction_values[:,i,j]).mean(0), 2)
    # plot_heatmap(interaction_arr)

    # 全局解释
    # shap_values = explainer.shap_values(X_test)
    # feature_importances=np.abs(shap_values).mean(1)[0]
    # plot_feature_importance_all(feature_importances,max_num_features=10)



    # plot_dot(X_test,shap_interaction_values[:,0,0])
    # plot_dot(X_test,shap_values[:,0])
    # print(explainer.expected_value)
    # print(np.mean(shap_values,axis=0))
    # print(np.abs(shap_values).mean(0))

    # shap.force_plot(explainer.expected_value[1], shap_values[1][0,:], X_test[0,:],matplotlib=True)

    # shap.summary_plot(shap_values, X_test)
    # shap.summary_plot(shap_values,plot_type="bar")
    # shap.dependence_plot(0,shap_values, X_test)

    # interaction_arr=np.zeros((X_test.shape[1],X_test.shape[1]))
    # for i in range(X_test.shape[1]):
    #     interaction_arr[i,:]=shap.utils.get_approximate_interactions(i,shap_values, X_test)
    # plot_heatmap(interaction_arr)
