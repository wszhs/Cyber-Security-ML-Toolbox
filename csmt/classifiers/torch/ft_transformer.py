
from collections import OrderedDict
from csmt.classifiers.abstract_model import AbstractModel
import torch
import torch.nn as nn
import numpy as np
from csmt.estimators.classification.pytorch import PyTorchClassifier
import torch.nn.functional as F
import random
from csmt.classifiers import rtdl

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

# def apply_model(x_num, x_cat=None):
#     return model(x_num, x_cat)

class FTTransformerTorch(AbstractModel):
    """
    FTTransformer.
    """

    def __init__(self, input_size,learning_rate=0.01,
                weight_decay=0,output_size=None):
        model = rtdl.FTTransformer.make_default(n_num_features=input_size,cat_cardinalities=None,last_layer_query_idx=[-1],d_out=output_size,)
        # model = rtdl.MLP.make_baseline(d_in=input_size,d_layers=[128, 256, 128],dropout=0.1,d_out=output_size,)
        criterion = nn.CrossEntropyLoss()
        # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        optimizer = model.make_default_optimizer()
        self.classifier = PyTorchClassifier(model=model,loss=criterion,optimizer=optimizer,input_shape=input_size,nb_classes=output_size)
