'''
Author: your name
Date: 2021-07-12 14:26:22
LastEditTime: 2021-07-13 16:15:33
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /Cyber-Security-ML-Toolbox/csmt/attacks/evasion/fast_gradient_zhs.py
'''
import numpy as np
from csmt.utils import (
    compute_success,
    get_labels_np_array,
    random_sphere,
    projection,
    check_and_transform_label_format,
)
class FastGradientZhsMethod():
    def __init__(self,estimator=None,norm=np.inf,eps=0.3,upper=1,lower=0):
        self.estimator=estimator
        self.norm=norm
        self.eps=eps
        self.upper=upper
        self.lower=lower
    def generate(self, x, y):
        y = check_and_transform_label_format(y, self.estimator.nb_classes)
        grad = self.estimator.loss_gradient(x, y)
        x_adv_path=np.zeros((x.shape[0],2,x.shape[1]))
        x_adv_path[:,0]=x
        if self.norm in [np.inf, "inf"]:
            grad = self.eps*np.sign(grad)
            grad = np.clip(x+grad, self.lower, self.upper)-x
        x_adv = x + grad
        x_adv_path[:,1]=x_adv
        return x_adv,x_adv_path