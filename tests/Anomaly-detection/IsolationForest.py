'''
Author: your name
Date: 2021-06-21 19:53:12
LastEditTime: 2021-07-15 17:53:02
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /Cyber-Security-ML-Toolbox/tests/Anomaly-detection/IsolationForest.py
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
from sklearn.preprocessing import MinMaxScaler
import sys
sys.path.append("/Users/zhanghangsheng/Documents/my_code/Cyber-Security-ML-Toolbox")
from csmt.get_model_data import get_datasets,parse_arguments,models_train,print_results,models_predict,models_and_ensemble_predict
import numpy as np


# filePath = 'csmt/datasets/data/Credit/creditcard.csv'
# df = pd.read_csv(filepath_or_buffer=filePath, header=0, sep=',')
# # print(df.shape[0])
# # print(df.head())

# df['Amount'] = StandardScaler().fit_transform(df['Amount'].values.reshape(-1, 1))
# df0 = df.query('Class == 0').sample(2000)
# df1 = df.query('Class == 1').sample(400)
# # print(df1)

# df = pd.concat([df0, df1])

# x_train, x_test, y_train, y_test = train_test_split(df.drop(labels=['Time', 'Class'], axis = 1) , 
#                                                     df['Class'], test_size=0.2, random_state=42)

if __name__=='__main__':
    arguments = sys.argv[1:]
    options = parse_arguments(arguments)
    datasets_name=options.datasets
    orig_models_name=options.algorithms
    
    X_train,y_train,X_val,y_val,X_test,y_test,n_features,mask=get_datasets(options)
    X_train=X_train[y_train==0]
    y_train=y_train[y_train==0]

    isolation_forest = IsolationForest(n_estimators=100, max_samples=256, contamination=0.1, random_state=42)

    isolation_forest.fit(X_train)

    anomaly_scores = isolation_forest.decision_function(X_test)
    anomaly_scores=anomaly_scores.reshape(-1,1)
    mm=MinMaxScaler()
    y_=mm.fit_transform(anomaly_scores)

    y_pred_=np.hstack((y_-0.2,1-y_))
    y_pred_=np.argmax(y_pred_, axis=1)
    print(y_pred_)



    # plt.figure(figsize=(15, 10))
    # plt.hist(anomaly_scores, bins=100)
    # plt.xlabel('Average Path Lengths', fontsize=14)
    # plt.ylabel('Number of Data Points', fontsize=14)
    # plt.show()

    from sklearn.metrics import roc_auc_score

    y_pred=[]
    for i in anomaly_scores:
        if i>0:
            y_pred.append(0)
        else:
            y_pred.append(1)
            
    auc = roc_auc_score(y_test, y_pred_)
    print("AUC: {:.2%}".format (auc))

    print(classification_report(y_test,y_pred_))
