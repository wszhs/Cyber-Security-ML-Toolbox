'''
Author: your name
Date: 2021-07-10 18:22:46
LastEditTime: 2021-07-10 18:48:25
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /Cyber-Security-ML-Toolbox/tests/interaction/test_dataset.py
'''
import sys
sys.path.append("/Users/zhanghangsheng/Documents/my_code/Cyber-Security-ML-Toolbox")
from csmt.datasets import load_kitsune

X,y=load_kitsune()
print(X[0:1])
print(y[0:1])