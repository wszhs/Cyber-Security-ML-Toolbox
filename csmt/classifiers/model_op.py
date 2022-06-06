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
import pickle

class ModelOperation():

    def __init__(self):
        pass

    def train(self,algorithm_name,datasets_name,model,if_adv,X_train,y_train,X_val,y_val):
        model.train(X_train,y_train,X_val,y_val)
        if if_adv==True:
            model_name=algorithm_name+'adv'+'.pkl'
        else:
            model_name=algorithm_name+'.pkl'
        finder_path=path.join('csmt/classifiers/saved_models',datasets_name)
        if path.exists(finder_path) is False:
            os.mkdir(finder_path)
        model_path=path.join('csmt/classifiers/saved_models',datasets_name,model_name)
        model.save(model_path)
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        return model

    def load(self,datasets_name,algorithm_name):
        #model_path
        model_name=algorithm_name+'.pkl'
        model_path=path.join('csmt/classifiers/saved_models',datasets_name,model_name)
        # load model
        with open(model_path, 'rb') as f:
            model=pickle.load(f)
        return model


    