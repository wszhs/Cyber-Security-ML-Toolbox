'''
Author: your name
Date: 2021-04-07 19:56:07
LastEditTime: 2021-04-07 21:07:55
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /Cyber-Security-ML-Toolbox/tests/test_sample.py
'''

import sys

from csmt.get_model_data import get_datasets,parse_arguments,train_models,load_models,print_results
import numpy as np
import math

if __name__=='__main__':

    
    arguments = sys.argv[1:]
    options = parse_arguments(arguments)
    X_train,y_train,X_test,y_test,n_features=get_datasets(options)
    
    action_distribute=[0.5,0.5]
    seg_point_arr=[]
    X_test_seg_arr=[]
    y_test_seg_arr=[]
    point=0
    seg_point_arr.append(0)
    for i in range(len(action_distribute)-1):
        point=point+math.floor(action_distribute[i]*len(X_test))
        seg_point_arr.append(point)
    seg_point_arr.append(len(X_test))
    print(seg_point_arr)

    for i in range(len(action_distribute)):
        X_test_seg_arr.append(X_test[seg_point_arr[i]:seg_point_arr[i+1]])
        y_test_seg_arr.append(y_test[seg_point_arr[i]:seg_point_arr[i+1]])


    