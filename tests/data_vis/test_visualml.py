
import sys
sys.path.append("/Users/zhanghangsheng/Documents/my_code/Cyber-Security-ML-Toolbox")
from csmt.get_model_data import get_datasets,parse_arguments,models_train,print_results,models_predict
import numpy as np
import torch
import random
from csmt.figure.visualml.visualml import decision_boundary_grid

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
# 设置随机数种子
setup_seed(20)
#torch.set_default_tensor_type(torch.DoubleTensor)

if __name__=='__main__':
     
     arguments = sys.argv[1:]
     options = parse_arguments(arguments)
     datasets_name=options.datasets
     orig_models_name=options.algorithms

     X_train,y_train,X_val,y_val,X_test,y_test,mask=get_datasets(options)
     X_train,y_train,X_test,y_test=X_train,y_train,X_test,y_test

     X_test_1=X_test[y_test==1]
     X_test_0=X_test[y_test==0]

     trained_models=models_train(datasets_name,orig_models_name,False,X_train,y_train,X_val,y_val)
     y_test,y_pred=models_predict(trained_models,X_test,y_test)

     table=print_results(datasets_name,orig_models_name,y_test,y_pred,'original_accuracy')

     decision_boundary_grid(trained_models[0],orig_models_name[0],X_test, y_test)



     
    
 









