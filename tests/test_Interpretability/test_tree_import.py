import sys
sys.path.append("/Users/zhanghangsheng/Documents/my_code/Cyber-Security-ML-Toolbox")
import matplotlib
import sage_zhs as sage

import matplotlib.pyplot as plt
import csmt.figure.visualml.visualml as vml
import pandas as pd
from csmt.get_model_data import get_datasets,parse_arguments,models_train,print_results,models_predict
import numpy as np
# from xgboost import plot_importance
from csmt.figure.visualml.plot_importance import plot_xg_importance,plot_feature_importance_all,plot_dot

if __name__=='__main__':
    arguments = sys.argv[1:]
    options = parse_arguments(arguments)
    datasets_name=options.datasets
    orig_models_name=options.algorithms
    evasion_algorithm=options.evasion_algorithm

    X_train,y_train,X_val,y_val,X_test,y_test,mask=get_datasets(options)

    trained_models=models_train(datasets_name,orig_models_name,False,X_train,y_train,X_val,y_val)

    y_test,y_pred=models_predict(trained_models,X_test,y_test)
    print_results(datasets_name,orig_models_name,y_test,y_pred,'original_accuracy')


    feature_importances=trained_models[0].classifier.model.feature_importances_
    print(feature_importances)
    plot_feature_importance_all(feature_importances,max_num_features=10)

    # plot_xg_importance(attack_model.classifier.model)
