'''
Author: your name
Date: 2021-06-24 14:11:40
LastEditTime: 2021-06-24 14:33:39
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /Cyber-Security-ML-Toolbox/tests/image/test_minst.py
'''
'''
Author: your name
Date: 2021-06-09 14:16:39
LastEditTime: 2021-06-11 20:46:55
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /Cyber-Security-ML-Toolbox/tests/evasion_attack/test_gradient_attack_self.py
'''
'''
Author: your name
Date: 2021-05-27 20:21:24
LastEditTime: 2021-06-07 15:54:41
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /Cyber-Security-ML-Toolbox/tests/test_art/test_rf_attack.py
'''
import sys
sys.path.append("/Users/zhanghangsheng/Documents/my_code/Cyber-Security-ML-Toolbox")
import sys
import keras
from keras import layers
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt

from csmt.estimators.classification.scikitlearn import SklearnClassifier
from csmt.datasets import load_mnist_flat_zhs,load_mnist_zhs

from sklearn import preprocessing
from csmt.classifiers.scores import get_binary_class_scores


X_train,y_train,X_test,y_test=load_mnist_zhs()
print(X_train.shape)

layers=[
    Conv2D(filters=4, kernel_size=(5, 5), strides=1, activation="relu", input_shape=(28, 28, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(filters=10, kernel_size=(5, 5), strides=1, activation="relu", input_shape=(23, 23, 4)),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(100, activation="relu"),
    Dense(10, activation="softmax")
]
model = keras.Sequential()
for layer in layers:
    model.add(layer)
print(model.summary())
model.compile(
    loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(lr=0.01), metrics=["accuracy"]
)

model.fit(X_train, y_train, batch_size=32, epochs=1)
y_pred = model.predict(X_test)
accuracy = np.sum(np.argmax(y_pred, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
print("Accuracy on benign test examples: {}%".format(accuracy * 100))
