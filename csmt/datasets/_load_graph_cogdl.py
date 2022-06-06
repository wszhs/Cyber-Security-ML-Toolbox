
import torch
import scipy.sparse as sp
import numpy as np

def normalize_feature(data):
    x_sum = torch.sum(data.x, dim=1)
    x_rev = x_sum.pow(-1).flatten()
    x_rev[torch.isnan(x_rev)] = 0.0
    x_rev[torch.isinf(x_rev)] = 0.0
    data.x = data.x * x_rev.unsqueeze(-1).expand_as(data.x)
    return data

def load_cora():
    data_file='csmt/datasets/data/cogdl/cora/data.pt'
    data=torch.load(data_file)
    data = normalize_feature(data)
    return data