import numpy as np
import os
import sys
sys.path.append("/Users/zhanghangsheng/Documents/my_code/Cyber-Security-ML-Toolbox")

from art.utils import load_diabetes

(x_train, y_train), (x_test, y_test), _, _ = load_diabetes(test_set=0.5)

from sklearn.linear_model import LinearRegression
from art.estimators.regression.scikitlearn import ScikitlearnRegressor

model = LinearRegression()
model.fit(x_train, y_train)

art_regressor = ScikitlearnRegressor(model)

print('Base model score: ', model.score(x_test, y_test))