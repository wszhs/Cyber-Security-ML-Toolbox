import sys
sys.path.append("/Users/zhanghangsheng/Documents/my_code/Cyber-Security-ML-Toolbox")
import matplotlib.pyplot as plt
import csmt.Interpretability.sage as sage
from sklearn.model_selection import train_test_split

import sklearn.linear_model as linear_model
import csmt.Interpretability.shap as shap
import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
from csmt.datasets import load_blob


X,y,mask=load_blob()
X_train,X_test,y_train,y_test = train_test_split(X,y, stratify=y,random_state=42)

model= linear_model.LogisticRegression(multi_class='ovr',max_iter=1000)
model.fit(X_train,y_train)

imputer = sage.MarginalImputer(model,X_test)
estimator = sage.PermutationEstimator(imputer, 'cross entropy')
# estimator = sage.IteratedEstimator(imputer, 'cross entropy')
# estimator = sage.KernelEstimator(imputer, 'cross entropy')
sage_values = estimator(X_test, y_test)

print(sage_values.values)



