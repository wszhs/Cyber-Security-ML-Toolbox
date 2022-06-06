'''
Author: your name
Date: 2021-07-05 16:12:43
LastEditTime: 2021-07-05 16:25:11
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /Cyber-Security-ML-Toolbox/tests/image/test.py
'''
'''
Author: your name
Date: 2021-07-04 19:23:38
LastEditTime: 2021-07-04 20:01:58
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /Cyber-Security-ML-Toolbox/tests/image/test.py
'''
import numpy as np
test_data = np.load('/Users/zhanghangsheng/Documents/my_code/Cyber-Security-ML-Toolbox/csmt/datasets/data/Kitsune/test.npy')
train_ben = np.load('/Users/zhanghangsheng/Documents/my_code/Cyber-Security-ML-Toolbox/csmt/datasets/data/Kitsune/train_ben.npy')
print(test_data[0:1])