'''
Author: your name
Date: 2021-07-07 19:19:20
LastEditTime: 2021-07-07 19:23:44
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /Cyber-Security-ML-Toolbox/tests/feature_extractor/test_read_npy.py
'''
import numpy as np
test_data = np.load('/Users/zhanghangsheng/Documents/my_code/Cyber-Security-ML-Toolbox/tests/example/Mirai.npy')
print(test_data[0:2])
print(test_data.shape)