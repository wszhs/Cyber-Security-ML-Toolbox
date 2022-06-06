'''
Author: your name
Date: 2021-03-24 19:23:21
LastEditTime: 2021-07-27 17:16:05
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /NashHE/nashhe/ml_ids/get_model_data.py
'''
# from csmt.utils import load_mnist
from collections import defaultdict
from random import choice
import sys
import numpy as np
import os
from os import path

from setuptools_scm import DEFAULT_VERSION_SCHEME
from csmt.attacks.attack import Attack
from csmt.get_data import get_graph_datasets,get_graph_cogdl_datasets,get_graph_grb_datasets
from csmt.get_graph_models import model_dict,models_train,models_predict,models_load

import pandas as pd
from pandas.core.frame import DataFrame

from csmt.classifiers.scores import get_class_scores
from sklearn.metrics import confusion_matrix
from tabulate import tabulate
import configargparse
import yaml
from sklearn import metrics


def parse_arguments(arguments):
    parser = configargparse.ArgParser(config_file_parser_class=configargparse.YAMLConfigFileParser)
    parser.add('--attack_models',required=False,default=['lr'],choices=['lr'])
    parser.add('--adv_train_models',required=False,default=['lr'],choices=['lr'])
    parser.add('--algorithms', required=False, default=['gcn'],choices=['mlp','gcn','graphsage','gat','gin'])
    parser.add('--datasets',required=False, default='grb-cora',choices=['grb-cora','cora','Weibo','Alpha','Elliptic'])
    parser.add('--evasion_algorithm',required=False,default=['random'],choices=['random','stack','flip'])
    parser.add('--adv_train_algorithm',required=False,default=['fgsm'],choices=['fgsm'])
    options = parser.parse_args(arguments)
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
    print(tabulate(rows, headers=headers))
    return rows_pandas
