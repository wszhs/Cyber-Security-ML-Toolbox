import sys
sys.path.append("/Users/zhanghangsheng/Documents/my_code/Cyber-Security-ML-Toolbox")
from csmt.get_model_data_graph import get_graph_grb_datasets,parse_arguments,models_train,models_predict,print_results,models_load
import numpy as np
import torch
import random
from csmt.graph_attack.evasion_attack import EvasionAttack

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

if __name__=='__main__':

     # 设置随机数种子
     setup_seed(20)
     arguments = sys.argv[1:]
     options = parse_arguments(arguments)
     datasets_name=options.datasets
     orig_models_name=options.algorithms
     evasion_algorithm=options.evasion_algorithm

     data=get_graph_grb_datasets(options)
     # print(data.index_test)
     # print(data.index_train)
     # print(data.index_val)

     # trained_models=models_train(datasets_name,orig_models_name,data)
     trained_models=models_load(datasets_name, orig_models_name)
     y_test,y_pred=models_predict(trained_models,data)
     table=print_results(datasets_name,orig_models_name,y_test,y_pred,'original_accuracy')

     adj_attack=EvasionAttack(evasion_algorithm,data)
     y_test,y_adv_pred=models_predict(trained_models,data,adj_attack)
     table=print_results(datasets_name,orig_models_name,y_test,y_adv_pred,'adversarial_accuracy')



