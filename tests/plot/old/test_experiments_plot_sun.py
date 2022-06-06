'''
Author: your name
Date: 2021-05-14 20:18:25
LastEditTime: 2021-06-27 14:43:56
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /Cyber-Security-ML-Toolbox/tests/test_experiments_plot.py
'''
from os import defpath
import sys
from numpy.core.fromnumeric import size
ROOT_PATH='/Users/zhanghangsheng/Documents/my_code/Cyber-Security-ML-Toolbox'
sys.path.append(ROOT_PATH)
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

def plot_fig10():
    df = pd.read_csv('experiments/plot/sun/anomaly.csv')
    x=[1,2,3,4]
    x_label = [1.5, 2.4, 3.6, 4.5]
    color=['#000000','#55321d','#a16742','#fda56e','#fdcd78']
    hatch=['','','','','']
    plt.figure(figsize=(10,6), dpi= 80)
    plt.bar(x=[i + 0.2 for i in x], height=df['0-0'], hatch=hatch[0],color=color[0], width=.14, label='(RFP,RFN)=(0,0)', alpha=1)
    plt.bar(x=[i + 0.35 for i in x], height=df['0.1-0.1'], color=color[1],hatch=hatch[1], width=.14, label='(RFP,RFN)=(0.1,0.1)', alpha=1)
    plt.bar(x=[i + 0.5 for i in x], height=df['0.2-0.2'], color=color[2],hatch=hatch[2], width=.14, label='(RFP,RFN)=(0.2,0.2)', alpha=1)
    plt.bar(x=[i + 0.65 for i in x], height=df['0.3-0.3'], color=color[3],hatch=hatch[3], width=.14, label='(RFP,RFN)=(0.3,0.3)', alpha=1)
    plt.bar(x=[i + 0.8 for i in x], height=df['0.4-0.4'], color=color[4],hatch=hatch[4],width=.14, label='(RFP,RFN)=(0.4,0.4)', alpha=1)

    plt.yticks(fontsize=25,rotation=0, horizontalalignment= 'right',)
    print(df['gca'] )
    plt.gca().set_xticks(x_label)
    plt.gca().set_xticklabels(df['gca'] , rotation=0, horizontalalignment= 'center',fontdict={'size':23})
    plt.legend(loc='upper left',ncol=1,bbox_to_anchor=(0.4,0.98),prop={'size':20},fancybox=True,shadow=True)
    plt.gca().set_ylabel('F1-Score',fontdict={'size':28})
    plt.gca().set_xlabel('',fontdict={'size':28}) 
    plt.show()
    # plt.savefig('experiments/figure/sun/loubaowubao.pdf',bbox_inches='tight', format='pdf', dpi=1000,transparent=True)

def plot_fig8():
    df = pd.read_csv('experiments/plot/sun/search.csv')
    x=[1,2,3,4,5]
    x_label = [1.5, 2.5, 3.5, 4.5, 5.5]
    color=['#000000','#55321d','#a16742','#fda56e','#fdcd78']
    hatch=['','','','','']
    plt.figure(figsize=(10,6), dpi= 80)
    plt.bar(x=[i + 0.2 for i in x], height=df['0.001-0.01'], hatch=hatch[0],color=color[0], width=.14, label='LTM=[0,0.01%)', alpha=1)
    plt.bar(x=[i + 0.35 for i in x], height=df['0.01-0.1'], color=color[1],hatch=hatch[1], width=.14, label='LTM=[0.01%,0.1%)', alpha=1)
    plt.bar(x=[i + 0.5 for i in x], height=df['0.1-1'], color=color[2],hatch=hatch[2], width=.14, label='LTM=[0.1%,1%)', alpha=1)
    plt.bar(x=[i + 0.65 for i in x], height=df['1-10'], color=color[3],hatch=hatch[3], width=.14, label='LTM=[1%,10%)', alpha=1)
    plt.bar(x=[i + 0.8 for i in x], height=df['10-100'], color=color[4],hatch=hatch[4],width=.14, label='LTM=[10%,100%)', alpha=1)

    plt.yticks(fontsize=25,rotation=0, horizontalalignment= 'right',)
    print(df['gca'] )
    plt.gca().set_xticks(x_label)
    plt.gca().set_xticklabels(df['gca'] , rotation=0, horizontalalignment= 'center',fontdict={'size':23})
    plt.legend(loc='upper left',ncol=1,bbox_to_anchor=(0.075,0.95),prop={'size':20},fancybox=True,shadow=True)
    plt.gca().set_ylabel('Time cost(S)',fontdict={'size':28})
    plt.gca().set_xlabel('',fontdict={'size':28}) 
    # plt.savefig('experiments/figure/sun/time_significance_err02_2.pdf',bbox_inches='tight', format='pdf', dpi=1000,transparent=True)
    plt.show()

def plot_fig7():
    df = pd.read_csv('experiments/plot/sun/search0.csv')
    x=[1,2,3,4,5]
    x_label = [1.5, 2.5, 3.5, 4.5, 5.5]
    color=['#000000','#55321d','#a16742','#fda56e','#fdcd78']
    # color=['#ffce98','#fda56e','#fb5f47','#e23329','#bc0f15']
    hatch=['','','','','']
    plt.figure(figsize=(10,6), dpi= 80)
    plt.bar(x=[i + 0.2 for i in x], height=df['0.001-0.01'], hatch=hatch[0],color=color[0], width=.14, label='LTM=[0,0.01%)', alpha=1)
    plt.bar(x=[i + 0.35 for i in x], height=df['0.01-0.1'], color=color[1],hatch=hatch[1], width=.14, label='LTM=[0.01%,0.1%)', alpha=1)
    plt.bar(x=[i + 0.5 for i in x], height=df['0.1-1'], color=color[2],hatch=hatch[2], width=.14, label='LTM=[0.1%,1%)', alpha=1)
    plt.bar(x=[i + 0.65 for i in x], height=df['1-10'], color=color[3],hatch=hatch[3], width=.14, label='LTM=[1%,10%)', alpha=1)
    plt.bar(x=[i + 0.8 for i in x], height=df['10-100'], color=color[4],hatch=hatch[4],width=.14, label='LTM=[10%,100%)', alpha=1)

    plt.yticks(fontsize=25,rotation=0, horizontalalignment= 'right',)
    print(df['gca'] )
    plt.gca().set_xticks(x_label)
    plt.gca().set_xticklabels(df['gca'] , rotation=0, horizontalalignment= 'center',fontdict={'size':23})
    plt.legend(loc='upper left',ncol=1,bbox_to_anchor=(0.4,0.95),prop={'size':20},fancybox=True,shadow=True)
    plt.gca().set_ylabel('F1-Score',fontdict={'size':28})
    plt.gca().set_xlabel('',fontdict={'size':28}) 
    plt.show()
    # plt.savefig('experiments/figure/sun/f1score_significance_outflow3.pdf', format='pdf',bbox_inches='tight', dpi=1000,transparent=True)

def plot_fig9():
    df = pd.read_csv('experiments/plot/sun/search1.csv')
    x=[1,2,3,4,5]
    x_label = [1.5, 2.5, 3.5, 4.5, 5.5]
    color=['#000000','#55321d','#fda56e','#fdcd78']
    hatch=['','','','','']
    plt.figure(figsize=(10,6), dpi= 100)
    plt.bar(x=[i + 0.2 for i in x], height=df['0.01'], hatch=hatch[0],color=color[0], width=.19, label='error=0.01%', alpha=1)
    plt.bar(x=[i + 0.4 for i in x], height=df['5'], color=color[1],hatch=hatch[1], width=.19, label='error=5%', alpha=1)
    plt.bar(x=[i + 0.6 for i in x], height=df['10'], color=color[2],hatch=hatch[2], width=.19, label='error=10%', alpha=1)
    plt.bar(x=[i + 0.8 for i in x], height=df['15'], color=color[3],hatch=hatch[3],width=.19, label='error=15%', alpha=1)

    plt.yticks(fontsize=25,rotation=0, horizontalalignment= 'right',)
    print(df['gca'] )
    plt.gca().set_xticks(x_label)
    plt.gca().set_xticklabels(df['gca'] , rotation=0, horizontalalignment= 'center',fontdict={'size':23})
    plt.legend(loc='upper left',ncol=1,bbox_to_anchor=(0.5,0.98),prop={'size':20},fancybox=True,shadow=True)
    plt.gca().set_ylabel('F1-Score',fontdict={'size':28})
    plt.gca().set_xlabel('',fontdict={'size':28}) 
    plt.show()
    # plt.savefig('experiments/figure/sun/f1score_predict_error_outflow.pdf', format='pdf',bbox_inches='tight',dpi=1000,transparent=True)

def plot_anomaly1():
    df = pd.read_csv('experiments/plot/sun/anomaly1.csv')
    x=[1,2,3,4,5,6]
    x_label = [1.5, 2.5, 3.5, 4.5, 5.5, 6.5]
    color=['#766f6e','#898787','#CCCCCC','#dfdfdf']
    hatch=['--','xx','//','\\\\']
    plt.figure(figsize=(10,6), dpi= 80)
    plt.bar(x=[i + 0.2 for i in x], height=df['iSwift'], hatch=hatch[0],color=color[0], width=.19, label='iSwift', alpha=1)
    plt.bar(x=[i + 0.4 for i in x], height=df['iSwift−wo−pruning'], color=color[1],hatch=hatch[1], width=.19, label='iSwift−wo−pruning', alpha=1)
    plt.bar(x=[i + 0.6 for i in x], height=df['iSwift−wo−filtering'], color=color[2],hatch=hatch[2], width=.19, label='iSwift−wo−filtering', alpha=1)
    plt.bar(x=[i + 0.8 for i in x], height=df['Aprior'], color=color[3],hatch=hatch[3], width=.19, label='Aprior', alpha=1)

    plt.yticks(fontsize=20,rotation=0, horizontalalignment= 'right',)
    print(df['gca'] )
    plt.gca().set_xticks(x_label)
    plt.gca().set_xticklabels(df['gca'] , rotation=0, horizontalalignment= 'center',fontdict={'size':19})
    plt.legend(loc='upper left',ncol=1,bbox_to_anchor=(0.58,0.95),prop={'size':16},fancybox=True,shadow=True)
    plt.gca().set_ylabel('F1-Score',fontdict={'size':20})
    plt.gca().set_xlabel('',fontdict={'size':18}) 
    plt.show()

if __name__=='__main__':
    plot_fig8()
