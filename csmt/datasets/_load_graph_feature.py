from scipy.io import loadmat
import pandas as pd
from csmt.datasets._base import get_mask,get_true_mask,add_str,get_dict
def load_amazon_feature():
    prefix='csmt/datasets/data/amazon_graph/'
    data = loadmat(prefix + 'Amazon.mat')
    X=data['features'].astype(float).toarray()
    y=data['label'].astype(int)[0]
    mask=get_true_mask([column for column in X])
    return X,y,mask

def load_yelp_feature():
    prefix='csmt/datasets/data/yelp_graph/'
    data = loadmat(prefix + 'YelpChi.mat')
    X=data['features'].astype(float).toarray()
    y=data['label'].astype(int)[0]
    mask=get_true_mask([column for column in X])
    return X,y,mask

def load_cora_feature():
    data_file='csmt/datasets/data/cogdl/cora/data.pt'
    import torch
    data=torch.load(data_file)
    X=data.x.numpy()
    y=data.y.numpy()
    mask=get_true_mask([column for column in X])
    return X,y,mask