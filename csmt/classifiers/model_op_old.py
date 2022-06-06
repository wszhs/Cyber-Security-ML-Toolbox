'''
Author: your name
Date: 2021-04-01 16:33:33
LastEditTime: 2021-08-05 09:43:15
LastEditors: your name
Description: In User Settings Edit
FilePath: /Cyber-Security-ML-Toolbox/csmt/classifiers/model_op.py
'''
'''
Author: your name
Date: 2021-03-24 16:41:46
LastEditTime: 2021-07-10 19:18:15
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /NashHE/nashhe/ml_ids/train/train_all_classic.py
'''
from tabulate import tabulate
import sys
import os
from os import path
import configargparse
import yaml
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
from csmt.get_models import model_dict

class ModelOperation():

    def __init__(self):
        pass

    def train(self,algorithm_name,datasets_name,model,if_adv,X_train,y_train):
        model.train(X_train,y_train)
        #model_path
        model_name=''
        with open(path.join('csmt/classifiers/config.yaml'),'r') as config_file:
            if if_adv==True:
                model_name=yaml.load(config_file.read(),Loader=yaml.FullLoader)[algorithm_name]['adv_model_name']
            else:
                model_name=yaml.load(config_file.read(),Loader=yaml.FullLoader)[algorithm_name]['model_name']
            finder_path=path.join('csmt/classifiers/saved_models',datasets_name)
            if path.exists(finder_path) is False:
                os.mkdir(finder_path)
            model_path=path.join('csmt/classifiers/saved_models',datasets_name,model_name)
        model.save(model_path)
        return model

    def load(self,algorithm_name,datasets_name,model):
        #model_path
        model_name=''
        with open(path.join('csmt/classifiers/config.yaml'),'r') as config_file:
            model_name=yaml.load(config_file.read(),Loader=yaml.FullLoader)[algorithm_name]['model_name']
        model_path=path.join('csmt/classifiers/saved_models',datasets_name,model_name)
        # load model
        model.load(model_path)
        return model


    