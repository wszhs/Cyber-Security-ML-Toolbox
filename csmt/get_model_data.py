
import sys
import numpy as np
import os
from os import path

from csmt.get_data import get_datasets,get_raw_datasets,get_graph_datasets
from csmt.get_models import model_dict,models_train,models_predict,models_predict_anomaly,models_load

import pandas as pd
from pandas.core.frame import DataFrame

from csmt.classifiers.scores import get_class_scores
from sklearn.metrics import confusion_matrix
from tabulate import tabulate
import configargparse
import yaml
from sklearn import metrics

# ['lr', 'svm','dt','rf','xgboost','lightgbm','catboost','deepforest','knn','hmm','mlp_keras','mlp_torch']
# ['kitnet','diff-rf','if','lr','dt','svm','mlp_torch','rf','xgboost','deepforest','soft_ensemble']
# ['androzoo','airbnb','bike','digits','cicids2018','drebin','blob','donut','kitsune','dohbrw','datacon','ctu13','satflow','nslkdd','cicids2017','contagiopdf','cicandmal2017','breast_cancer','iris_zhs','mnist_flat','mnist']
# ['gradient','swarm','de','zo_shap_sgd','zo_shap_scd','zoo','tree','nes','random','zones','zoscd','zosgd','zoadamm','zoadamm_sum','zones_adamm_sum','mimicry','zosgd_sum','zosgd_shap_sum']
# 'hard_ensemble','soft_ensemble','stacking_ensemble'
# 'lr','dt','svm','mlp_torch','xgboost','ft_transformer','hard_ensemble','soft_ensemble','stacking_ensemble'
# 异常检测模型
# if diff-rf kitnet ae ocsvm hbos vae
# grad_free openbox_opt bayes
# cora_graph yelp_graph

def parse_arguments(arguments):
......
    return options
 
def print_results(datasets_name,models_name,y_test,y_pred_arr,label):
    headers = ['datasets','algorithm','accuracy', 'f1', 'precision', 'recall','roc_auc','ASR']
    rows=[]
    for i in range(0,len(models_name)):
        y_pred=np.argmax(y_pred_arr[i], axis=1)
        result=get_class_scores(y_test, y_pred)
        row=list(result)
        row.insert(0,models_name[i])
        row.insert(0,datasets_name)
        rows.append(row)
    rows_pandas=DataFrame(rows)
    rows_pandas.columns=headers
    finder_path=path.join('experiments/plot/',datasets_name)
    if path.exists(finder_path) is False:
        os.mkdir(finder_path)
    rows_pandas.to_csv('experiments/plot/'+datasets_name+'/'+label+'.csv',index=False)
    print(tabulate(rows, headers=headers))

    # 针对二分类
    # print_ASR_ALL(models_name,y_test,y_pred_arr)
    # print_ASR_AVG(models_name,y_test,y_pred_arr)
    # print_DSR_ALL(models_name,y_test,y_pred_arr)
    return rows_pandas

def print_ASR_ALL(models_name,y_test,y_pred_arr):
    #提取黑样本
    y_test_1=y_test[y_test==1]
    y_pred_arr_1=y_pred_arr[:,y_test==1]

    K=len(models_name)
    adv_maps = np.full((K,len(y_test_1)), False)
    for k in range(K):
        y_pred=np.argmax(y_pred_arr_1[k], axis=1)
        adv_maps[k]=(y_pred != y_test_1)
    # print(adv_maps)
    asr_all = np.full(len(y_test_1), True)
    for adv_map in adv_maps:
        # print (np.sum(asr_all))
        asr_all = np.logical_and(adv_map, asr_all)
    print ('ASR_all: %.2f %%' % (100 * np.sum(asr_all) / float(len(y_test_1))))

def print_ASR_AVG(models_name,y_test,y_pred_arr):
    #提取黑样本
    y_test_1=y_test[y_test==1]
    y_pred_arr_1=y_pred_arr[:,y_test==1]
    K=len(models_name)
    r_k=0
    for k in range(K):
        r_k+= (1-metrics.recall_score(y_test_1, np.argmax(y_pred_arr_1[k], axis=1)))*100
    r_k=r_k/K
    print ('ASR_avg: %.2f %%' % (r_k))

def print_DSR_ALL(models_name,y_test,y_pred_arr):
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
    print ('DSR_all: %.2f %%' % (100 * np.sum(dsr_all) / float(len(y_test_1))))

def print_results_ensemble(datasets_name,models_name,y_test,y_pred_arr,label):
    headers = ['datasets','algorithm','accuracy', 'f1', 'precision', 'recall','roc_auc']
    rows=[]
    for i in range(0,len(models_name)):
        y_pred=np.argmax(y_pred_arr[i], axis=1)
        result=get_class_scores(y_test, y_pred)
        row=list(result)
        row.insert(0,models_name[i])
        row.insert(0,datasets_name)
        rows.append(row)
    rows_pandas=DataFrame(rows)
    rows_pandas.columns=headers
    rows_pandas.to_csv('experiments/plot/'+datasets_name+'/'+label+'.csv',index=False)
    print(label)
    print(tabulate(rows, headers=headers))

