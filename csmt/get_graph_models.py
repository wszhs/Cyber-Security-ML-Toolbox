
from csmt.classifiers.graph.model.torch import GCN
from csmt.classifiers.graph.model.torch import MLP
from csmt.classifiers.graph.model.torch import GraphSAGE
from csmt.classifiers.graph.model.torch import GIN
from csmt.classifiers.graph.model.dgl import GAT
from csmt.classifiers.graph.utils.normalize import SAGEAdjNorm
from csmt.classifiers.graph.utils.normalize import GCNAdjNorm
from csmt.classifiers.graph.trainer.trainer import Trainer
import csmt.classifiers.graph.utils as utils
import torch
import numpy as np
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def model_dict(algorithm,data):
    models_dic={
        'mlp':MLP(in_features=data.num_features,
            out_features=data.num_classes,
            hidden_features=64,
            n_layers=3),
        'gcn':GCN(in_features=data.num_features,
               out_features=data.num_classes,
               hidden_features=64, 
               n_layers=3,
               adj_norm_func=GCNAdjNorm,
               layer_norm=True,
               residual=False,
               dropout=0.5),
        'graphsage': GraphSAGE(in_features=data.num_features,
                  out_features=data.num_classes,
                  hidden_features=64,
                  n_layers=3,
                  adj_norm_func=SAGEAdjNorm,
                  layer_norm=False,
                  dropout=0.5),
        'gin': GIN(in_features=data.num_features,
            out_features=data.num_classes,
            hidden_features=64, 
            n_layers=3,
            adj_norm_func=None,
            layer_norm=False,
            batch_norm=True,
            dropout=0.5),
        'gat':GAT(in_features=data.num_features,
            out_features=data.num_classes,
            hidden_features=64,
            n_layers=3,
            n_heads=4,
            adj_norm_func=None,
            layer_norm=False,
            residual=False,
            feat_dropout=0.6,
            attn_dropout=0.6,
            dropout=0.5)
    }

    return models_dic[algorithm]

def get_model(algorithms_name,data):
    models_array=[]
    for i in range(len(algorithms_name)):
        models_array.append(model_dict(algorithms_name[i],data))
    return models_array,algorithms_name

def models_train(datasets_name,models_name,data):
    models_array,algorithms_name=get_model(models_name,data)
    trained_models_array=[]
    for i in range(0,len(models_array)):
        model = models_array[i].to(device)
        save_dir = "csmt/classifiers/saved_models/{}/{}".format(datasets_name, models_name[i])
        save_name = "model.pt"
        train_mode = "inductive"  # "transductive"

        trainer = Trainer(dataset=data, 
                    optimizer=torch.optim.Adam(model.parameters(), lr=0.01),
                    loss=torch.nn.functional.cross_entropy,
                    lr_scheduler=False,
                    early_stop=True,
                    early_stop_patience=500,
                    feat_norm=None,
                    device=device)

        trainer.train(model=model, 
               n_epoch=2000,
               eval_every=1,
               save_after=0,
               save_dir=save_dir,
               save_name=save_name,
               train_mode=train_mode,
               verbose=False)
        trained_models_array.append(model)
    return trained_models_array

def models_load(datasets_name,models_name):
    models_array=[]
    for i in range(0,len(models_name)):
        model = torch.load(os.path.join('csmt/classifiers/saved_models',datasets_name,models_name[i],'model.pt'))
        model = model.to(device)
        models_array.append(model)
    return models_array

def models_predict(trained_models,data,adj=None):
    y_pred_arr=np.zeros((len(trained_models),data.num_test,data.num_classes))
    for i in range(len(trained_models)):
        if adj==None:
            y_test, y_pred = utils.evaluate(trained_models[i], 
                                        features=data.features,
                                        adj=data.adj,
                                        labels=data.labels,
                                        feat_norm=None,
                                        adj_norm_func=trained_models[i].adj_norm_func,
                                        mask=data.test_mask,
                                        device=device)
        else:
            y_test, y_pred = utils.evaluate(trained_models[i], 
                            features=data.features,
                            adj=adj,
                            labels=data.labels,
                            feat_norm=None,
                            adj_norm_func=trained_models[i].adj_norm_func,
                            mask=data.test_mask,
                            device=device)
        y_pred_arr[i]=y_pred
    return y_test,y_pred_arr