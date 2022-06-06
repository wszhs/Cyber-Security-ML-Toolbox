'''
Author: your name
Date: 2021-07-12 09:59:00
LastEditTime: 2021-08-02 08:52:34
LastEditors: your name
Description: In User Settings Edit
FilePath: /Cyber-Security-ML-Toolbox/csmt/classifiers/ensemble/ensemble.py
'''

from csmt.classifiers.classic.decision_tree import DecisionTree
from csmt.classifiers.classic.k_nearest_neighbours import KNearestNeighbours
from csmt.classifiers.classic.logistic_regression import LogisticRegression
from csmt.classifiers.classic.random_forest import RandomForest
from csmt.classifiers.classic.support_vector_machine import SupportVectorMachine
from csmt.classifiers.classic.naive_bayes import NaiveBayes
from csmt.classifiers.classic.xgboost import XGBoost
from csmt.classifiers.classic.lightgbm import LightGBM
from csmt.classifiers.classic.catboost import CatBoost
from csmt.classifiers.classic.deepforest import DeepForest
from csmt.classifiers.classic.hidden_markov_model import HMM
from csmt.classifiers.torch.mlp import MLPTorch,MLP2Torch
from csmt.classifiers.torch.cnn import CNNTorch
from csmt.classifiers.keras.mlp import MLPKeras
from csmt.classifiers.torch.lr import LRTorch
from csmt.classifiers.torch.cnn import CNNTorch,CNNMnistTorch
from csmt.classifiers.abstract_model import AbstractModel
from csmt.estimators.classification.ensemble_tree import EnsembleTree
from csmt.estimators.classification.soft_ensemble import SoftEnsemble
from csmt.estimators.classification.hard_ensemble import HardEnsemble
from csmt.estimators.classification.stacking_ensemble import StackingEnsemble
from csmt.estimators.classification.bayes_ensemble import BayesEnsemble
from csmt.estimators.classification.origin_ensemble import OriginEnsemble
from csmt.classifiers.anomaly_detection.KitNET import KitNET
from csmt.classifiers.anomaly_detection.IsolationForest import IsolationForest
from csmt.classifiers.anomaly_detection.diff_rf import DIFFRF

def model_dict(algorithm,n_features,out_size):
    models_dic={
        'lr':LogisticRegression,
        'knn':KNearestNeighbours,
        'dt':DecisionTree,
        'nb':NaiveBayes,
        'svm':SupportVectorMachine,
        'hmm':HMM,
        'rf':RandomForest,
        'xgboost':XGBoost,
        'lightgbm':LightGBM,
        'catboost':CatBoost,
        'deepforest':DeepForest,
        'mlp_torch':MLPTorch,
        'mlp2_torch':MLP2Torch,
        'lr_torch':LRTorch,
        'cnn_torch':CNNTorch,
        'cnn_mnist_torch':CNNMnistTorch,
        'mlp_keras':MLPKeras,
        'kitnet':KitNET,
        'if':IsolationForest,
        'diff-rf':DIFFRF
    }
    return models_dic[algorithm](input_size=n_features,output_size=out_size)


class HardEnsembleModel(AbstractModel):
    def __init__(self,input_size,output_size):
        models_name=['mlp_torch','mlp2_torch']
        models=[]
        for i in range(len(models_name)):
            model=model_dict(models_name[i],input_size,output_size)
            models.append(model)
        self.classifier=HardEnsemble(model=models,nb_features=input_size, nb_classes=output_size,clip_values=(0,1))

class SoftEnsembleModel(AbstractModel):
    def __init__(self,input_size,output_size):
        models_name=['dt','xgboost']
        models=[]
        for i in range(len(models_name)):
            model=model_dict(models_name[i],input_size,output_size)
            models.append(model)
        self.classifier=SoftEnsemble(model=models,nb_features=input_size, nb_classes=output_size,clip_values=(0,1))

class StackingEnsembleModel(AbstractModel):
    def __init__(self,input_size,output_size):
        models_name=['mlp_torch','xgboost']
        models=[]
        for i in range(len(models_name)):
            model=model_dict(models_name[i],input_size,output_size)
            models.append(model)
        self.classifier=StackingEnsemble(model=models,nb_features=input_size, nb_classes=output_size,clip_values=(0,1))

class BayesEnsembleModel(AbstractModel):
    def __init__(self,input_size,output_size):
        models_name=['lr','dt']
        models=[]
        for i in range(len(models_name)):
            model=model_dict(models_name[i],input_size,output_size)
            models.append(model)
        self.classifier=BayesEnsemble(model=models,nb_features=input_size, nb_classes=output_size,clip_values=(0,1))

class TransferEnsembleModel(AbstractModel):
    def __init__(self,models_name,input_size,output_size):
        if 'bayes_' in models_name:
            models_name=models_name.replace('bayes_','')
        self.models_name=models_name.replace('transfer_','').split("+")
        models=[]
        for i in range(len(self.models_name)):
            model=model_dict(self.models_name[i],input_size,output_size)
            models.append(model)
        self.classifier=OriginEnsemble(model=models,nb_features=input_size, nb_classes=output_size,clip_values=(0,1))
