'''
Author: your name
Date: 2021-04-17 10:04:22
LastEditTime: 2021-04-17 10:58:36
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /Cyber-Security-ML-Toolbox/tests/test_hmm.py
'''
import sys
sys.path.append("/Users/zhanghangsheng/Documents/my_code/Cyber-Security-ML-Toolbox")
from csmt.get_model_data import get_datasets,parse_arguments,train_models,load_models,print_results
import numpy as np

from hmmlearn import hmm
import numpy as np
from hmm_classifier import HMM_classifier

if __name__=='__main__':

    
    arguments = sys.argv[1:]
    options = parse_arguments(arguments)
    
    X_train,y_train,X_test,y_test,n_features=get_datasets(options)

    # print(x[0])
    # print(x.shape)
    X_train=np.reshape(X_train,(X_train.shape[0],X_train.shape[1],1))
    X_test=np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))
    print(X_train[0])
    print(X_test[0])

    model = HMM_classifier(hmm.GaussianHMM())
    model.fit(X_train,y_train)

    # # Predict probability per label
    # pred = model.predict_proba(X_test[0])
    # print(pred)

    # # Get label with the most high probability
    for i in range(0,10):
        pred = model.predict(X_test[i])
        print(y_test[i])
        print(pred)