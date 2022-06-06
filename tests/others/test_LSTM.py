'''
Author: your name
Date: 2021-04-17 11:06:44
LastEditTime: 2021-04-19 11:21:55
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /Cyber-Security-ML-Toolbox/tests/test_LSTM.py
'''
import sys
sys.path.append("/Users/zhanghangsheng/Documents/my_code/Cyber-Security-ML-Toolbox")
from csmt.get_model_data import get_datasets,models_train,parse_arguments,models_train,print_results,models_predict,attack_models_train
from keras.models import Sequential
from keras.layers import LSTM, Dense
import numpy as np
import math

arguments = sys.argv[1:]
options = parse_arguments(arguments)
# X_train,y_train,X_test,y_test,n_features=get_datasets(options)
X_train,y_train,X_val,y_val,X_test,y_test,mask=get_datasets(options)

X_train=np.reshape(X_train,(X_train.shape[0],X_train.shape[1],1))
X_test=np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))

data_dim = 1
timesteps = X_train.shape[1]
batch_size = 1
epochs = 1

# 期望输入数据尺寸: (batch_size, timesteps, data_dim)
# 请注意，我们必须提供完整的 batch_input_shape，因为网络是有状态的。
# 第 k 批数据的第 i 个样本是第 k-1 批数据的第 i 个样本的后续。
model = Sequential()
model.add(LSTM(32, return_sequences=True, stateful=True,
               batch_input_shape=(batch_size, timesteps, data_dim)))
model.add(LSTM(32, return_sequences=True, stateful=True))
model.add(LSTM(32, stateful=True))
model.add(Dense(2, activation='softmax'))
for layer in model.layers:
    print(layer.output_shape)

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


for i in range(epochs):
    print("Epoch {:d}/{:d}".format(i+1, epochs))
    model.fit(X_train, y_train, batch_size=batch_size, epochs=1, validation_data=(X_test, y_test), shuffle=True)
    model.reset_states()

score, _ = model.evaluate(X_test, y_test, batch_size=batch_size)      # 返回误差值和度量值
rmse = math.sqrt(score)
print("\nMSE: {:.3f}, RMSE: {:.3f}".format(score, rmse))

pre = model.predict(X_test, batch_size=batch_size)