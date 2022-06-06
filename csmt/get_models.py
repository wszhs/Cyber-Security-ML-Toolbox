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
from csmt.classifiers.keras.lstm import LSTMKeras
from csmt.classifiers.torch.mlp import MLPTorch,MLP2Torch
from csmt.classifiers.torch.lr import LRTorch
from csmt.classifiers.torch.ft_transformer import FTTransformerTorch
from csmt.classifiers.torch.cnn import CNNTorch,CNNMnistTorch,LeNetMnistTorch
from csmt.classifiers.keras.mlp import MLPKeras
from csmt.classifiers.anomaly_detection.KitNET import KitNET
from csmt.classifiers.anomaly_detection.IsolationForest import IsolationForest
from csmt.classifiers.anomaly_detection.other_anomaly import *
from csmt.classifiers.anomaly_detection.autoencoder import AbAutoEncoder
from csmt.classifiers.anomaly_detection.diff_rf import DIFFRF
from csmt.classifiers.ensemble.ensemble import SoftEnsembleModel
from csmt.classifiers.ensemble.ensemble import HardEnsembleModel
from csmt.classifiers.ensemble.ensemble import StackingEnsembleModel
from csmt.classifiers.ensemble.ensemble import BayesEnsembleModel
from csmt.classifiers.ensemble.ensemble import TransferEnsembleModel

from csmt.classifiers.model_op import ModelOperation
import numpy as np

from csmt.estimators.classification.hard_ensemble import HardEnsemble

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
        'ft_transformer':FTTransformerTorch,
        'cnn_torch':CNNTorch,
        'cnn_mnist_torch':CNNMnistTorch,
        'lenet_mnist_torch':LeNetMnistTorch,
        'mlp_keras':MLPKeras,
        'lstm_keras':LSTMKeras,
        'kitnet':KitNET,
        'ae':AbAutoEncoder,
        'if':IsolationForest,
        'ocsvm':AbOCSVM,
        'hbos':AbHBOS,
        'vae':AbVAE,
        'diff-rf':DIFFRF,
        'soft_ensemble':SoftEnsembleModel,
        'hard_ensemble':HardEnsembleModel,
        'stacking_ensemble':StackingEnsembleModel,
        'bayes_ensemble':BayesEnsembleModel
    }
    return models_dic[algorithm](input_size=n_features,output_size=out_size)

def get_model(algorithms_name,n_features,out_size):
    models_array=[]
    for i in range(len(algorithms_name)):
        models_array.append(model_dict(algorithms_name[i],n_features,out_size))
    return models_array,algorithms_name

def models_train(datasets_name,models_name,if_adv,X_train,y_train,X_val,y_val):
    out_size=len(np.unique(y_train))
    models_array,algorithms_name=get_model(models_name,X_train.shape[1],out_size)
    trained_models_array=[]
    for i in range(0,len(models_array)):
        model_=ModelOperation()
        trained_model=model_.train(algorithms_name[i],datasets_name,models_array[i],if_adv,X_train,y_train,X_val,y_val)
        trained_models_array.append(trained_model)
    return trained_models_array

def models_load(datasets_name,models_name):
    models_array=[]
    for i in range(0,len(models_name)):
        model_=ModelOperation()
        trained_model=model_.load(datasets_name,models_name[i])
        models_array.append(trained_model)
    return models_array

def models_predict_anomaly(trained_models,X_test,y_test):
    if len(np.unique(y_test))<=2:
        len_y=2
    else:
        len_y=len(np.unique(y_test))
    y_pred_arr=np.zeros((len(trained_models),X_test.shape[0],len_y))
    for i in range(len(trained_models)):
        trained_models[i].predict_anomaly(X_test,y_test)
    
def models_predict(trained_models,X_test,y_test):
    if len(np.unique(y_test))<=2:
        len_y=2
    else:
        len_y=len(np.unique(y_test))
    y_pred_arr=np.zeros((len(trained_models),X_test.shape[0],len_y))
    for i in range(len(trained_models)):
        y_pred=trained_models[i].predict(X_test)
        y_pred_arr[i]=y_pred
    return y_test,y_pred_arr
