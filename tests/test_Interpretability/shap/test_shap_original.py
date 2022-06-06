import sys
sys.path.append("/Users/zhanghangsheng/Documents/my_code/Cyber-Security-ML-Toolbox")
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import sklearn.linear_model as linear_model
from sklearn import tree
import csmt.Interpretability.shap as shap
import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
from csmt.datasets import load_blob


X,y,mask=load_blob()
X_train,X_test,y_train,y_test = train_test_split(X,y, stratify=y,random_state=42)

model= linear_model.LogisticRegression(multi_class='ovr',max_iter=1000)
# model = tree.DecisionTreeClassifier()
model.fit(X_train,y_train)

explainer = shap.Explainer(model,X_train)
shap_values = explainer(X_test)
shap.summary_plot(shap_values,plot_type="bar")