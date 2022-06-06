'''
Author: your name
Date: 2021-04-23 12:23:36
LastEditTime: 2021-04-23 15:03:16
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /Cyber-Security-ML-Toolbox/tests/test_data_pre.py
'''
import sys
ROOT_PATH='/Users/zhanghangsheng/Documents/my_code/Cyber-Security-ML-Toolbox'
sys.path.append(ROOT_PATH)

import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import LabelEncoder

file_path='csmt/datasets/data/CTU-13-log/ctu-13_all.csv'
df = pd.read_csv(file_path, encoding='utf8', low_memory=False)

print(df.shape)

vec = DictVectorizer()
df = pd.read_csv(file_path, encoding='utf8', low_memory=False)

#映射为数组
# vs1 = list(map(lambda x: {'label': x}, df['label']))
# vs2 = vec.fit_transform(vs1).toarray()
# for index, name in enumerate(vec.get_feature_names()):
#     df[name] = list(map(lambda x: int(x[index]), vs2))

#映射为数字
# labelencoder = LabelEncoder()
# df['sni'] = labelencoder.fit_transform(df['sni'])

dict_val=df['client_ciphers'].value_counts().items()
dic_={}
i=0
for key,value in dict_val:
    i=i+1
    dic_[key]=i
print(dic_)

def add_str(x):
    x_arr=x.split(',')
    while len(x_arr) < 20:
        x_arr.append('0')
    x_out = ','.join(x_arr)
    return x_out

df['resp_spl']=df['resp_spl'].map(lambda x:add_str(str(x)))
print(df['resp_spl'])

for i in range(20):
    df['resp_spl_'+str(i)]=df['resp_spl'].map(lambda x:str(x).split(',')[i])






