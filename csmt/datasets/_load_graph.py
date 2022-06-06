from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import pickle
from typing import Tuple
import scipy.io as sio
import pandas as pd
import numpy as np
import scipy.sparse as sp
from os import path
from csmt.datasets._base import get_mask,get_true_mask,add_str,get_dict
from csmt.datasets.graph.utils import preprocess
from scipy.io import loadmat
import torch
import scipy.sparse as sp
import numpy as np

class Pyg2Dpr():
    def __init__(self, pyg_data):
        n = pyg_data.num_nodes
        self.adj = sp.csr_matrix((np.ones(pyg_data.edge_index[0].shape[0]),
            (pyg_data.edge_index[0], pyg_data.edge_index[1])), shape=(n, n))
        self.features = pyg_data.x.numpy()
        self.labels = pyg_data.y.numpy()


def normalize_feature(data):
    x_sum = torch.sum(data.x, dim=1)
    x_rev = x_sum.pow(-1).flatten()
    x_rev[torch.isnan(x_rev)] = 0.0
    x_rev[torch.isinf(x_rev)] = 0.0
    data.x = data.x * x_rev.unsqueeze(-1).expand_as(data.x)
    return data


def get_train_val_test(nnodes, val_size=0.1, test_size=0.8, stratify=None, seed=None):
    """This setting follows nettack/mettack, where we split the nodes
    into 10% training, 10% validation and 80% testing data
    """
    assert stratify is not None, 'stratify cannot be None!'
    if seed is not None:
        np.random.seed(seed)

    idx = np.arange(nnodes)
    train_size = 1 - val_size - test_size
    idx_train_and_val, idx_test = train_test_split(idx,
                                                   random_state=None,
                                                   train_size=train_size + val_size,
                                                   test_size=test_size,
                                                   stratify=stratify)
    if stratify is not None:
        stratify = stratify[idx_train_and_val]

    idx_train, idx_val = train_test_split(idx_train_and_val,
                                          random_state=None,
                                          train_size=(train_size / (train_size + val_size)),
                                          test_size=(val_size / (train_size + val_size)),
                                          stratify=stratify)
    return idx_train, idx_val, idx_test

def load_npz(file_name):
    with np.load(file_name) as loader:
        # loader = dict(loader)
        adj = sp.csr_matrix((loader['adj_data'], loader['adj_indices'],
                                    loader['adj_indptr']), shape=loader['adj_shape'])
        if 'attr_data' in loader:
            features = sp.csr_matrix((loader['attr_data'], loader['attr_indices'],
                                            loader['attr_indptr']), shape=loader['attr_shape'])
        else:
            features = None
        labels = loader.get('labels')
    if features is None:
        features = np.eye(adj.shape[0])
    features = sp.csr_matrix(features, dtype=np.float32)
    return adj, features, labels

def get_adj():
    adj, features, labels = load_npz('/tmp/cora.npz')
    adj = adj + adj.T
    adj = adj.tolil()
    adj[adj > 1] = 1
    # whether to set diag=0?
    adj.setdiag(0)
    adj = adj.astype("float32").tocsr()
    adj.eliminate_zeros()
    assert np.abs(adj - adj.T).sum() == 0, "Input graph is not symmetric"
    assert adj.max() == 1 and len(np.unique(adj[adj.nonzero()].A1)) == 1, "Graph must be unweighted"
    return adj, features, labels

def load_cora_graph_1():
    adj, features, labels=get_adj()
    idx_train, idx_val, idx_test=get_train_val_test(nnodes=adj.shape[0], val_size=0.1, test_size=0.8, stratify=labels, seed=20)
    split_ids = [idx_train, idx_val, idx_test]
    # print(type(features), type(adj), type(labels), type(split_ids))
    return features, adj, labels, split_ids

def load_cora_graph_2():
    data = Dataset(root='/tmp/', name='cora')
    adj, features, labels = data.adj, data.features, data.labels
    idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
    adj, features, labels = preprocess(adj, features, labels, preprocess_adj=False, sparse=True)
    split_ids = [idx_train, idx_val, idx_test]
    return features, adj, labels, split_ids

def load_cora_graph():
    data_file='csmt/datasets/data/cogdl/cora/data.pt'
    data=torch.load(data_file)
    data = normalize_feature(data)
    data=Pyg2Dpr(data)
    features=data.features
    adj=data.adj
    labels=data.labels
    idx_train, idx_val, idx_test=get_train_val_test(nnodes=adj.shape[0], val_size=0.1, test_size=0.8, stratify=labels, seed=20)
    split_ids = [idx_train, idx_val, idx_test]
    return features, adj, labels, split_ids

def load_alpha_graph():
    data_file='csmt/datasets/data/cogdl/alpha/data.pt'
    data=torch.load(data_file)
    data = normalize_feature(data)
    data=Pyg2Dpr(data)
    features=data.features
    adj=data.adj
    labels=data.labels
    idx_train, idx_val, idx_test=get_train_val_test(nnodes=adj.shape[0], val_size=0.1, test_size=0.8, stratify=labels, seed=20)
    split_ids = [idx_train, idx_val, idx_test]
    return features, adj, labels, split_ids

def load_elliptic_graph():
    data_file='csmt/datasets/data/cogdl/elliptic/data.pt'
    data=torch.load(data_file)
    data = normalize_feature(data)
    data=Pyg2Dpr(data)
    features=data.features
    adj=data.adj
    labels=data.labels
    idx_train, idx_val, idx_test=get_train_val_test(nnodes=adj.shape[0], val_size=0.1, test_size=0.8, stratify=labels, seed=20)
    split_ids = [idx_train, idx_val, idx_test]
    return features, adj, labels, split_ids

def load_weibo_graph():
    data_file='csmt/datasets/data/cogdl/Weibo/data.pt'
    data=torch.load(data_file)
    data = normalize_feature(data)
    data=Pyg2Dpr(data)
    features=data.features
    adj=data.adj
    labels=data.labels
    idx_train, idx_val, idx_test=get_train_val_test(nnodes=adj.shape[0], val_size=0.1, test_size=0.8, stratify=labels, seed=20)
    split_ids = [idx_train, idx_val, idx_test]
    return features, adj, labels, split_ids

def load_yelp_graph(path: str = 'csmt/datasets/data/yelp_graph/YelpChi.mat',
                   train_size: int = 0.8, meta: bool = False) -> \
        Tuple[list, np.array, list, np.array]:
    """
    The data loader to load the Yelp heterogeneous information network data
    source: http://odds.cs.stonybrook.edu/yelpchi-dataset

    :param path: the local path of the dataset file
    :param train_size: the percentage of training data
    :param meta: if True: it loads a HIN with three meta-graphs,
                 if False: it loads a homogeneous rur meta-graph
    """
    data = sio.loadmat(path)
    truelabels, features = data['label'], data['features'].astype(float)
    truelabels = truelabels.tolist()[0]

    if not meta:
        rownetworks = [data['net_rur']]
    else:
        rownetworks = [data['net_rur'], data['net_rsr'], data['net_rtr']]

    y = truelabels
    index = np.arange(len(y))
    X_train, X_test, y_train, y_test = train_test_split(index,
                                                        y,
                                                        stratify=y,
                                                        test_size=1-train_size,
                                                        random_state=48,
                                                        shuffle=True)
    X_train, X_val, y_train, y_val = train_test_split(X_train,
                                                      y_train,
                                                      stratify=y_train,
                                                      test_size=0.2,
                                                      random_state=48,
                                                      shuffle=True)

    split_ids = [X_train, y_train, X_val, y_val, X_test, y_test]
    adj=rownetworks[0]

    # print(type(features), type(adj), type(np.array(y)), type(split_ids))

    return features, adj, np.array(y), split_ids

def load_amazon_graph():
    train_size=0.8
    prefix='csmt/datasets/data/amazon_graph/'
    data = sio.loadmat(prefix + 'Amazon.mat')
    truelabels, features = data['label'], data['features'].astype(float)
    truelabels = truelabels.tolist()[0]

    y = truelabels
    index = np.arange(len(y))
    rownetworks = [data['homo']]

    X_train, X_test, y_train, y_test = train_test_split(index,
                                                        y,
                                                        stratify=y,
                                                        test_size=1-train_size,
                                                        random_state=48,
                                                        shuffle=True)
    X_train, X_val, y_train, y_val = train_test_split(X_train,
                                                      y_train,
                                                      stratify=y_train,
                                                      test_size=0.2,
                                                      random_state=48,
                                                      shuffle=True)

    split_ids = [X_train, y_train, X_val, y_val, X_test, y_test]
    adj=rownetworks[0]

    # print(type(features), type(adj), type(np.array(y)), type(split_ids))
    labels=np.array(y).astype(int)
    return features, adj, labels, split_ids






