'''
Author: your name
Date: 2021-05-15 10:12:11
LastEditTime: 2021-05-15 11:11:21
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /Cyber-Security-ML-Toolbox/tests/data_vis/test_vis.py
'''
import sys
sys.path.append("/Users/zhanghangsheng/Documents/my_code/Cyber-Security-ML-Toolbox")
import pandas as pd 
import numpy as np 
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt 
import seaborn as sns 
from csmt.datasets import load_datacon

def plot_density(df):
    # # Draw Plot
    plt.figure(figsize=(13,10), dpi= 80)
    sns.distplot(df.loc[df['label'] == 0, "flow_duration"], color="dodgerblue", label="white", hist_kws={'alpha':.7}, kde_kws={'linewidth':3})
    sns.distplot(df.loc[df['label'] == 1, "flow_duration"], color="orange", label="black", hist_kws={'alpha':.7}, kde_kws={'linewidth':3})
    plt.ylim(0, 0.15)
    plt.legend()
    # plt.savefig('densityH.pdf', format='pdf', dpi=1000,transparent=True)
    plt.show()

def plot_pairwise(df):
    # df=df[['fwd_pkts_tot','bwd_pkts_tot','fwd_data_pkts_tot','bwd_data_pkts_tot','label']]
    df=df[['fwd_data_pkts_tot','fwd_data_pkts_tot','label']]
    print(df)
    plt.figure(figsize=(10,8), dpi= 80)
    sns.pairplot(df, kind="scatter", hue="label")
    # plt.savefig('pairwise.pdf', format='pdf', dpi=1000,transparent=True)
    plt.show()

def plot_marginal_histogram(df):
    # Create Fig and gridspec
    fig = plt.figure(figsize=(16, 10), dpi= 80)
    grid = plt.GridSpec(4, 4, hspace=0.5, wspace=0.2)

    # Define the axes
    ax_main = fig.add_subplot(grid[:-1, :-1])
    ax_right = fig.add_subplot(grid[:-1, -1], xticklabels=[], yticklabels=[])
    ax_bottom = fig.add_subplot(grid[-1, 0:-1], xticklabels=[], yticklabels=[])

    # Scatterplot on main ax
    ax_main.scatter('flow_duration', 'label', c=df.label.astype('category').cat.codes, alpha=.9, data=df,cmap="tab10", edgecolors='gray', linewidths=.5)

    # histogram on the right
    ax_bottom.hist(df.label, 40, histtype='stepfilled', orientation='vertical', color='orange')
    ax_bottom.invert_yaxis()
    # histogram in the bottom
    ax_right.hist(df.label, 40, histtype='stepfilled', orientation='horizontal', color='darkcyan')


    # Decorations
    # ax_main.set(title='Scatterplot with Histograms \n displ vs hwy', xlabel='displ', ylabel='hwy')
    ax_main.title.set_fontsize(20)
    for item in ([ax_main.xaxis.label, ax_main.yaxis.label] + ax_main.get_xticklabels() + ax_main.get_yticklabels()):
        item.set_fontsize(14)

    xlabels = ax_main.get_xticks().tolist()
    ax_main.set_xticklabels(xlabels)
    # plt.savefig('marginH.pdf', format='pdf', dpi=1000,transparent=True)
    plt.show()

def plot_marginal_boxplot(df):
    fig = plt.figure(figsize=(16, 10), dpi= 80)
    grid = plt.GridSpec(4, 4, hspace=0.5, wspace=0.2)

    # Define the axes
    ax_main = fig.add_subplot(grid[:-1, :-1])
    ax_right = fig.add_subplot(grid[:-1, -1], xticklabels=[], yticklabels=[])
    ax_bottom = fig.add_subplot(grid[-1, 0:-1], xticklabels=[], yticklabels=[])

    # Scatterplot on main ax
    ax_main.scatter('flow_duration', 'label', c=df.label.astype('category').cat.codes, alpha=.9, data=df,cmap="tab10", edgecolors='gray', linewidths=.5)

        # Add a graph in each part
    sns.boxplot(y=df.flow_duration, ax=ax_right, orient='v')
    sns.boxplot(df.flow_duration, ax=ax_bottom, orient="h")

    # Decorations
    # ax_main.set(title='Scatterplot with Histograms \n displ vs hwy', xlabel='displ', ylabel='hwy')
    ax_main.title.set_fontsize(20)
    for item in ([ax_main.xaxis.label, ax_main.yaxis.label] + ax_main.get_xticklabels() + ax_main.get_yticklabels()):
        item.set_fontsize(14)

    xlabels = ax_main.get_xticks().tolist()
    ax_main.set_xticklabels(xlabels)
    # plt.savefig('marginH.pdf', format='pdf', dpi=1000,transparent=True)
    plt.show()


if __name__=='__main__':
    X,y,mask=load_datacon()
    df=pd.concat([X,y], axis=1)
    print(df)
    # plot_density(df)
    # plot_pairwise(df)
    # plot_marginal_histogram(df)
    plot_marginal_boxplot(df)


    