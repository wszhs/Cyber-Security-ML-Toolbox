'''
Author: your name
Date: 2021-06-21 14:59:44
LastEditTime: 2021-06-22 14:17:28
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /Cyber-Security-ML-Toolbox/tests/KitNET/Autoencoder.py
'''
import sys

from keras import layers
sys.path.append("/Users/zhanghangsheng/Documents/my_code/Cyber-Security-ML-Toolbox")
import keras
from keras import optimizers
from keras import losses
from keras.models import Sequential, Model
from keras.layers import Dense, Input, Dropout, Embedding, LSTM,Conv2D,MaxPooling2D,Flatten
from keras.layers import Convolution1D, Dense, Dropout, Flatten, MaxPooling1D
from keras.layers import LSTM, SimpleRNN, GRU
from keras.optimizers import RMSprop, Adam, Nadam
from keras.preprocessing import sequence
from keras.callbacks import TensorBoard
from keras import regularizers

import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report


import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import seaborn as sns

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import tensorflow 
import sys

from sklearn.model_selection import train_test_split
from csmt.datasets import load_contagiopdf
from sklearn import preprocessing
# print("Python: ", sys.version)

# print("pandas: ", pd.__version__)
# print("numpy: ", np.__version__)
# print("seaborn: ", sns.__version__)
# print("matplotlib: ", matplotlib.__version__)
# print("sklearn: ", sklearn.__version__)
# print("Keras: ", keras.__version__)
# print("Tensorflow: ", tensorflow.__version__)


# filePath = 'csmt/datasets/data/Credit/creditcard.csv'
# df = pd.read_csv(filepath_or_buffer=filePath, header=0, sep=',')

# df['Amount'] = StandardScaler().fit_transform(df['Amount'].values.reshape(-1, 1))
# df0 = df.query('Class == 0').sample(20000)
# df1 = df.query('Class == 1').sample(400)
# df = pd.concat([df0, df1])
# X_train, X_test, y_train, y_test = train_test_split(df.drop(labels=['Time', 'Class'], axis = 1) , 
#                                                     df['Class'], test_size=0.2, random_state=42)
# X_train, X_test, y_train, y_test=X_train.values, X_test.values, y_train.values, y_test.values
# print(X_train.shape[0])

X,y=load_contagiopdf()
scaler1 = preprocessing.MinMaxScaler().fit(X)
X = scaler1.transform(X)
X_train,X_test,y_train,y_test = train_test_split(X,y, stratify=y,train_size=0.8, test_size=0.2,random_state=42)
y_train,y_test=y_train.values,y_test.values

layers = [
    LSTM(32,return_sequences=True,input_shape=(X_train.shape[1],1)),
    LSTM(32),
    Dense(2, activation='softmax')
]

# layers = [
#     GRU(32,input_shape=(X_train.shape[1],1)),
#     Dense(2, activation='sigmoid')
# ]

# layers = [
#     SimpleRNN(32,input_shape=(X_train.shape[1],1)),
#     Dense(2, activation='sigmoid')
# ]

# layers=[
#     Convolution1D(64,3,activation='relu',input_shape=(X_train.shape[1],1)),
#     MaxPooling1D(pool_size=2),
#     Flatten(),
#     Dense(128,activation='relu'),
#     Dropout(0.5),
#     Dense(2,activation='sigmoid')
# ]

# layers=[
#     Conv2D(32,(3,3),activation='relu',input_shape=(27,5,1)),
#     Conv2D(32,(3,3),activation='relu'),
#     Flatten(),
#     Dense(128,activation='relu'),
#     Dense(2,activation='sigmoid')
# ]

# layers = [
#     Dense(128, activation='relu',input_shape=(X_train.shape[1],)),
#     Dense(64,activation='relu'),
#     Dense(32,activation='relu'),
#     Dense(16,activation='relu'),
#     Dense(2,activation='sigmoid')
#     ]

# layers=[
#     Conv2D(32, kernel_size=(5, 5), strides=(1, 1),
#                  activation='relu',
#                  input_shape=X_train.shape[1])
# ]
   
model = keras.Sequential()
for layer in layers:
    model.add(layer)
print(model.summary())
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
X_train=np.reshape(X_train,(X_train.shape[0],X_train.shape[1],1))
# X_train=np.reshape(X_train,(X_train.shape[0],27,5))
# X_train=np.reshape(X_train,(X_train.shape[0],27,5,1))
print(X_train.shape)
model.fit(X_train, y_train, batch_size=32, epochs=10)