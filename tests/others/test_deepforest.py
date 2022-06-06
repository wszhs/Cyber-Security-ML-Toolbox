'''
Author: your name
Date: 2021-04-17 10:59:57
LastEditTime: 2021-04-17 11:04:27
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /Cyber-Security-ML-Toolbox/tests/test_deepforest.py
'''
import sys
sys.path.append("/Users/zhanghangsheng/Documents/my_code/Cyber-Security-ML-Toolbox")
from csmt.get_model_data import get_datasets,parse_arguments,train_models,load_models,print_results
import numpy as np
import numpy as np
from deepforest import CascadeForestClassifier
if __name__=='__main__':

    
    arguments = sys.argv[1:]
    options = parse_arguments(arguments)
    
    X_train,y_train,X_test,y_test,n_features=get_datasets(options)

    #deepforest
    model = CascadeForestClassifier(random_state=1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pro=model.predict_proba(X_test)
    print(y_pro)