'''
Author: your name
Date: 2021-05-14 18:44:09
LastEditTime: 2021-05-19 13:24:22
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /Cyber-Security-ML-Toolbox/csmt/plot.py
'''


import matplotlib
matplotlib.use('TkAgg')
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import numpy as np
import pandas as pd

def plot_transfer(table,datasets_name):
    matplotlib.style.use('seaborn-whitegrid')
    df = table
    plt.figure(figsize=(21,9), dpi= 80)
    y_label=df['attack'].values
    df=df.drop(['attack'], axis=1)
    h = sns.heatmap(df/100,vmin=0,vmax=1, xticklabels = df.columns, yticklabels = y_label, cmap='copper',
                        annot=True,annot_kws={'size':22},cbar=False)

    cb=h.figure.colorbar(h.collections[0]) #显示colorbar
    cb.ax.tick_params(labelsize=26) #设置colorbar刻度字体大小。

    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22,rotation=0, horizontalalignment= 'right')
    plt.show()
    # plt.savefig('experiments/figure/'+datasets_name+'_'+'trans1.pdf', format='pdf', dpi=1000,transparent=True)

def plot_roc(models_name,y_test,y_pred_arr):
    NUMBER=len(models_name)
    values = range(NUMBER)
    y_test_arr=[]
    jet = plt.get_cmap('jet') 
    cNorm  = colors.Normalize(vmin=0, vmax=values[-1])
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
    colors_arr=[]

    for idx in range(len(y_pred_arr)):
        y_test_arr.append(y_test)
        colorVal = scalarMap.to_rgba(values[idx])
        colors_arr.append(colorVal)

    lines_arr = [':','--','-.','-']
    linestyles=np.random.choice(lines_arr, size=NUMBER, replace=True)
    plt.figure(figsize=(10,6))
    # Seaborn's beautiful styling
    sns.set_style('darkgrid', {'axes.facecolor': '0.9'})
    for model_name,y_test,y_pred,clr,ls in zip(models_name,y_test_arr,y_pred_arr,colors_arr,linestyles):
        fpr,tpr,trhresholds = roc_curve(y_true=y_test,y_score=y_pred)
        roc_auc = auc(x=fpr,y=tpr)
        plt.plot(fpr,tpr,color=clr,linestyle=ls,label='%s (auc=%0.2f)'%(model_name,roc_auc))
    lw = 2
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.legend(loc='lower right')
    plt.plot([0,1],[0,1],linestyle='--',color='gray',linewidth=2)
    plt.yticks([i/20.0 for i in range(21)])
    plt.xticks([i/20.0 for i in range(21)])
    plt.xlabel('False positive rate (FPR)')
    plt.ylabel('True positive rate (TPR)')
    plt.savefig('experiments/figure/roc.pdf', format='pdf', dpi=1000,transparent=True)
    # plt.show()

def plot_line(res,keys):

    table_list=[]
    for i in range(len(res)):
        row_x=[res[i]['params'][key] for key in keys]
        row_y=res[i]['target']
        row_x.append(row_y)
        table_list.append(row_x)
    keys.append('target')
    df=pd.DataFrame(table_list,columns=keys)
    keys.pop()

    plt.figure(figsize=(20,5), dpi= 80)
    df['target'].plot(kind='line')
    plt.show()
