'''
Author: your name
Date: 2021-06-21 14:43:51
LastEditTime: 2021-06-21 14:43:52
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /Cyber-Security-ML-Toolbox/tests/KitNET/utils.py
'''
import numpy
numpy.seterr(all='ignore')


def sigmoid(x):
    return 1. / (1 + numpy.exp(-x))


def dsigmoid(x):
    return x * (1. - x)

def tanh(x):
    return numpy.tanh(x)

def dtanh(x):
    return 1. - x * x

def softmax(x):
    e = numpy.exp(x - numpy.max(x))  # prevent overflow
    if e.ndim == 1:
        return e / numpy.sum(e, axis=0)
    else:  
        return e / numpy.array([numpy.sum(e, axis=1)]).T  # ndim = 2


def ReLU(x):
    return x * (x > 0)

def dReLU(x):
    return 1. * (x > 0)