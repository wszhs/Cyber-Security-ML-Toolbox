import sys
sys.path.append("/Users/zhanghangsheng/Documents/my_code/Cyber-Security-ML-Toolbox")
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from sklearn import tree
import xgboost
import seaborn as sns

import csmt.Interpretability.shap as shap
import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
from csmt.datasets import load_blob
from csmt.figure.visualml.plot_importance import plot_xg_importance,plot_feature_importance_all,plot_dot

def plot_heatmap(table):
    matplotlib.style.use('seaborn-whitegrid')
    # table = np.random.rand(10, 12)
    sns.heatmap(table,vmin=0,vmax=1, cmap='viridis',annot=True)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14,rotation=0, horizontalalignment= 'right')
    plt.show()

X,y,mask=load_blob()
X_train,X_test,y_train,y_test = train_test_split(X,y, stratify=y,random_state=42)

# model = tree.DecisionTreeClassifier()
model=xgboost.XGBClassifier()
model.fit(X_train,y_train)

explainer = shap.TreeExplainer(model, X_train)
shap_values = explainer.shap_values(X_test)
# plot_dot(X_test,y_test)


# shap.force_plot(explainer.expected_value, shap_values[0,:], X_test[0,:],matplotlib=True)

# 全局解释
# shap.summary_plot(shap_values,plot_type="bar")
# shap.summary_plot(shap_values, X_test)
# 依赖图
# shap.dependence_plot(0,shap_values, X_test)

# 树博弈交互值
# shap_interaction_values=explainer.shap_interaction_values(X_test)
# shap.summary_plot(shap_interaction_values, max_display=10)

# interaction_arr=np.zeros((X_test.shape[1],X_test.shape[1]))
# for i in range(X_test.shape[1]):
#     for j in range(X_test.shape[1]):
#         interaction_arr[i,j]=round(np.abs(shap_interaction_values[:,i,j]).mean(0), 2)
# plot_heatmap(interaction_arr)

# plot_dot(X_test,shap_interaction_values[:,0,0])
# plot_dot(X_test,shap_values[:,0])
# print(explainer.expected_value)

# 近似夏普利交互值
# interaction_arr=np.zeros((X_test.shape[1],X_test.shape[1]))
# for i in range(X_test.shape[1]):
#     interaction_arr[i,:]=shap.utils.approximate_interactions(i,shap_values, X_test)
# plot_heatmap(interaction_arr)


