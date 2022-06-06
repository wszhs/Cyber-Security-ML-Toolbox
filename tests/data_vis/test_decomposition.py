'''
Author: your name
Date: 2021-05-14 17:05:47
LastEditTime: 2021-07-20 14:54:43
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /Cyber-Security-ML-Toolbox/tests/data_vis/test_TSNE.py
'''
import sys
sys.path.append("/Users/zhanghangsheng/Documents/my_code/Cyber-Security-ML-Toolbox")
import pandas as pd 
import numpy as np 
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.preprocessing import MinMaxScaler
from csmt.get_model_data import get_datasets,parse_arguments
from csmt.figure.visualml.plot_ds import plot_ds_2d,plot_ds_3d
 
# df= pd.read_csv('csmt/datasets/data/others/mushrooms.csv')
# print(df.head())

# X = df.drop('class', axis=1) 
# y = df['class'] 
# y = y.map({'p': 'Posionous', 'e': 'Edible'}) 
# print(y)

arguments = sys.argv[1:]
options = parse_arguments(arguments)
datasets_name=options.datasets
orig_models_name=options.algorithms
X_train,y_train,X_val,y_val,X_test,y_test,mask=get_datasets(options)
X,y=X_train,y_train

# from sklearn.preprocessing import StandardScaler 
# from sklearn.decomposition import PCA 
 
# X_std = StandardScaler().fit_transform(X) 
# X_pca = PCA(n_components=2).fit_transform(X_std) 
# X_pca = np.vstack((X_pca.T, y)).T 
 
# df_pca = pd.DataFrame(X_pca, columns=['1st_Component','2nd_Component', 'class']) 
# df_pca.head()

# plt.figure(figsize=(8, 8)) 
# sns.scatterplot(data=df_pca, hue='class', x='1st_Component', y='2nd_Component') 
# plt.show()

from sklearn.manifold import TSNE 
 
tsne = TSNE(n_components=2) 
X_tsne = tsne.fit_transform(X) 
mm=MinMaxScaler()
X_tsne=mm.fit_transform(X_tsne)
X_tsne_data = np.vstack((X_tsne.T, y)).T 
df_tsne = pd.DataFrame(X_tsne_data, columns=['Dim1', 'Dim2', 'class']) 
df_tsne.head()

# plt.figure(figsize=(8, 8)) 
# sns.scatterplot(data=df_tsne, hue='class', x='Dim1', y='Dim2') 
# plt.savefig('experiments/figure/tsne.pdf', format='pdf', dpi=1000,transparent=True)
# plt.show()

# plot_ds_2d(X,y)