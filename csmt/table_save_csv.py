'''
Author: your name
Date: 2021-05-18 00:24:01
LastEditTime: 2021-05-18 09:45:58
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /Cyber-Security-ML-Toolbox/csmt/table_save_csv.py
'''
import numpy as np
import pandas as pd
from scipy.optimize import moduleTNC

def save_nash_csv(index,res,keys,datasets_name,label):
    table_list=[]
    for i in range(len(res)):
        row_x=[res[i]['params'][key] for key in keys]
        row_y=res[i]['target']
        row_x.append(row_y)
        table_list.append(row_x)
    keys.append('target')
    df=pd.DataFrame(table_list,columns=keys)
    if index==0:
        df.to_csv('experiments/plot/'+datasets_name+'/'+label+'.csv',index=False)
    else:
        df.to_csv('experiments/plot/'+datasets_name+'/'+label+'.csv',index=False,mode='a',header=False)
