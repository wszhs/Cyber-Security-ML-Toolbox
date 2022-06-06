'''
Author: your name
Date: 2021-06-21 20:17:23
LastEditTime: 2021-06-21 20:25:20
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /Cyber-Security-ML-Toolbox/tests/Anomaly-detection/ocsvm.py
'''
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.svm import OneClassSVM

filePath = 'csmt/datasets/data/Credit/creditcard.csv'
df = pd.read_csv(filepath_or_buffer=filePath, header=0, sep=',')
# print(df.shape[0])
# print(df.head())

df['Amount'] = StandardScaler().fit_transform(df['Amount'].values.reshape(-1, 1))
df0 = df.query('Class == 0').sample(2000)
df1 = df.query('Class == 1').sample(400)
# print(df1)

df = pd.concat([df0, df1])

x_train, x_test, y_train, y_test = train_test_split(df.drop(labels=['Time', 'Class'], axis = 1) , 
                                                    df['Class'], test_size=0.2, random_state=42)

ocsvm = OneClassSVM(kernel='rbf', gamma=0.00005, nu=0.1)

ocsvm.fit(x_train)

# y_pred = ocsvm.predict(x_test)
anomaly_scores = ocsvm.decision_function(x_test)
# plt.figure(figsize=(15, 10))
# plt.hist(anomaly_scores, bins=100)
# plt.xlabel('Average Path Lengths', fontsize=14)
# plt.ylabel('Number of Data Points', fontsize=14)
# plt.show()

from sklearn.metrics import roc_auc_score

y_pred=[]
for i in anomaly_scores:
    if i>-2:
        y_pred.append(0)
    else:
        y_pred.append(1)
auc = roc_auc_score(y_test, y_pred)
print("AUC: {:.2%}".format (auc))

print(classification_report(y_test,y_pred))

