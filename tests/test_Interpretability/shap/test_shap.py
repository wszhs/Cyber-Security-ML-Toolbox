import sys
sys.path.append("/Users/zhanghangsheng/Documents/my_code/Cyber-Security-ML-Toolbox")
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import xgboost
import shap
import numpy as np
import pandas as pd 
import xgboost as xgb
import xgboost as xgb
from sklearn.model_selection import train_test_split
from csmt.datasets import load_blob
X,y,mask=load_blob()
X_train,X_test,y_train,y_test = train_test_split(X,y, stratify=y,random_state=42)

d_train = xgb.DMatrix(X_train, label=y_train)
d_test = xgb.DMatrix(X_test, label=y_test)

params = {
    "eta": 0.01,
    "objective": "binary:logistic",
    "subsample": 0.5,
    "base_score": np.mean(y_train),
    "eval_metric": "logloss"
}
model = xgb.train(params, d_train, 5000, evals = [(d_test, "test")], verbose_eval=100, early_stopping_rounds=20)

# this takes a minute or two since we are explaining over 30 thousand samples in a model with over a thousand trees
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_train)
# shap.force_plot(explainer.expected_value, shap_values[0,:], X_train[0,:])
shap.summary_plot(shap_values,plot_type="bar")