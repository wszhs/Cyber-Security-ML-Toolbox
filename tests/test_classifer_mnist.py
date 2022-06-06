
import sys
from catboost import train
sys.path.append("/Users/zhanghangsheng/Documents/my_code/Cyber-Security-ML-Toolbox")
from csmt.get_model_data import get_datasets,parse_arguments,models_train,print_results,models_predict,models_load
import numpy as np
import torch
import matplotlib.pyplot as plt
from csmt.Interpretability.captum.attr import *
import csmt.Interpretability.quantus as quantus
import random
import pandas as pd

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
# 设置随机数种子
setup_seed(20)

if __name__=='__main__':
     
     arguments = sys.argv[1:]
     options = parse_arguments(arguments)
     datasets_name=options.datasets
     orig_models_name=options.algorithms

     X_train,y_train,X_val,y_val,X_test,y_test,mask=get_datasets(options)
     X_train,y_train,X_test,y_test=X_train,y_train,X_test,y_test


     # trained_models=models_train(datasets_name,orig_models_name,False,X_train,y_train,X_val,y_val)
     trained_models=models_load(datasets_name,orig_models_name)
     y_test,y_pred=models_predict(trained_models,X_test,y_test)

     table=print_results(datasets_name,orig_models_name,y_test,y_pred,'original_accuracy')




     
    
 









