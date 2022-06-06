import sys
import string

sys.path.append("/Users/zhanghangsheng/Documents/my_code/Cyber-Security-ML-Toolbox")
from csmt.get_model_data import get_datasets,parse_arguments,models_train,print_results,models_predict,models_load
import numpy as np


def train():
    arguments = sys.argv[1:]
    options = parse_arguments(arguments)
    datasets_name=options.datasets
    orig_models_name=options.algorithms
    
    X_train,y_train,X_val,y_val,X_test,y_test,mask=get_datasets(options)

    trained_models=models_train(datasets_name,orig_models_name,False,X_train,y_train,X_val,y_val)
    y_test,y_pred=models_predict(trained_models,X_test,y_test)
    
    table=print_results(datasets_name,orig_models_name,y_test,y_pred,'original_accuracy')

    return trained_models

def load():
    arguments = sys.argv[1:]
    options = parse_arguments(arguments)
    datasets_name=options.datasets
    orig_models_name=options.algorithms
    X_train,y_train,X_val,y_val,X_test,y_test,mask=get_datasets(options)
    
    models=models_load(datasets_name,orig_models_name)

    y_test,y_pred=models_predict(models,X_test,y_test)
    table=print_results(datasets_name,orig_models_name,y_test,y_pred,'original_accuracy')


if __name__=='__main__':
    model=train()
    # model=load()




