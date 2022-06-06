'''
Author: your name
Date: 2021-05-14 20:18:25
LastEditTime: 2021-05-20 22:49:01
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /Cyber-Security-ML-Toolbox/tests/test_experiments_plot.py
'''
import sys
from numpy.core.fromnumeric import size
ROOT_PATH="/Users/yuanlu/Desktop/画图/实验图/代码"
sys.path.append(ROOT_PATH)
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
matplotlib.style.use('seaborn-whitegrid')
import numpy as np

def plot_feature_evasion():
    df_flow = pd.read_csv('experiments/plot/kitsune/original_accuracy.csv')
    df_flow_adv = pd.read_csv('experiments/plot/kitsune/adversarial_accuracy.csv')

    df_spl = pd.read_csv('experiments/plot/kitsune/original_accuracy.csv')
    df_spl_adv = pd.read_csv('experiments/plot/kitsune/adversarial_accuracy.csv')

    df_tls = pd.read_csv('experiments/plot/kitsune/original_accuracy.csv')
    df_tls_adv = pd.read_csv('experiments/plot/kitsune/adversarial_accuracy.csv')

    df_all = pd.read_csv('experiments/plot/kitsune/original_accuracy.csv')
    df_all_adv = pd.read_csv('experiments/plot/kitsune/adversarial_accuracy.csv')

    n = df_flow['algorithm'].unique().__len__()

    # colors = [plt.cm.RdBu_r(i/float(n*5)) for i in range(n)]
    # colors2 = [plt.cm.RdBu(i/float(n*4)) for i in range(n)]

    # plt.figure(figsize=(4,3), dpi= 80)
    # plt.subplots_adjust(left=0.05, bottom=None, right=0.95, top=None,
    #                 wspace=0.1, hspace=None)
    x = [1]
    for i in range(n-1):
        x.append(x[i]+1.5)

    plt.figure(figsize=(35,6), dpi= 80)

    plt.bar(x=[i - 0.45 for i in x],height=df_flow['recall'], color='#ab7332', width=.25, label='FLOW',alpha=0.7)
    plt.bar(x=[i - 0.45 for i in x],height=df_flow_adv['recall'], color='#ab7332', width=.25, label='FLOW(M-TBA)',alpha=1)

    plt.bar(x=[i - 0.15 for i in x], height=df_spl['recall'], color='#2f2216', width=.25, label='SPL', alpha=0.6)
    plt.bar(x=[i - 0.15 for i in x], height=df_spl_adv['recall'], color='#2f2216', width=.25, label='SPL(M_TBA)', alpha=1)

    plt.bar(x=[i + 0.15 for i in x], height=df_tls['recall'], color='#935c3a', width=.25, label='TLS', alpha=0.7)
    plt.bar(x=[i + 0.15 for i in x], height=df_tls_adv['recall'], color='#935c3a', width=.25, label='TLS(M_TBA)', alpha=1)
    
    plt.bar(x=[i + 0.45 for i in x], height=df_all['recall'], color='#fec47d', width=.25, label='ALL', alpha=0.7)
    plt.bar(x=[i + 0.45 for i in x], height=df_all_adv['recall'], color='#e6752e', width=.25, label='ALL(M_TBA)', alpha=1)

    plt.yticks(fontsize=20,rotation=0, horizontalalignment= 'right',)
    
    
    plt.gca().set_xticks(x)
    plt.gca().set_xticklabels(df_flow['algorithm'] , rotation=0, horizontalalignment= 'center',fontdict={'size':20})
    # plt.title("adv_train vs orig", fontsize=14)
    plt.legend(loc='upper center',ncol=4, borderaxespad=-4,prop={'size':13})
    plt.gca().set_ylabel('Adversarial Detection Rate',fontdict={'size':22})
    plt.gca().set_xlabel('Machine Learning Model',fontdict={'size':22}) 
    
    # plt.subplot(141)
    # plt.bar(df_flow['algorithm'], df_flow['recall'], color=colors, width=.5, alpha=.8)
    # plt.bar(df_flow_adv['algorithm'], df_flow_adv['recall'], color=colors2, width=.5, alpha=1)
    # plt.xticks([])
    # plt.yticks(fontsize=14,rotation=0, horizontalalignment= 'right')

    # plt.subplot(142)
    # plt.bar(df_spl['algorithm'], df_spl['recall'], color=colors, width=.5, alpha=.8)
    # plt.bar(df_spl_adv['algorithm'], df_spl_adv['recall'], color=colors2, width=.5, alpha=1)
    # plt.xticks([])
    # plt.yticks(fontsize=14,rotation=0, horizontalalignment= 'right')
    # plt.title("SPLT", fontsize=14)

    # plt.subplot(143)
    # plt.bar(df_tls['algorithm'], df_tls['recall'], color=colors, width=.5, alpha=.8)
    # plt.bar(df_tls_adv['algorithm'], df_tls_adv['recall'], color=colors2, width=.5, alpha=1)
    # plt.xticks([])
    # plt.yticks(fontsize=14,rotation=0, horizontalalignment= 'right')
    # plt.title("TLS", fontsize=14)

    # plt.subplot(144)
    # plt.bar(df_all['algorithm'], df_all['recall'], color=colors, width=.5, alpha=.8)
    # plt.bar(df_all_adv['algorithm'], df_all_adv['recall'], color=colors2, width=.5, alpha=1)
    # plt.xticks([])
    # plt.yticks(fontsize=14,rotation=0, horizontalalignment= 'right')
    # plt.title("ALL", fontsize=14)
    plt.show()
    # plt.savefig('experiments/figure/cicandmal2017_feature_evasion1.pdf', format='pdf', dpi=1000,transparent=True)

def plot_dataset_evasion():
    df_cicandmal2017 = pd.read_csv('experiments/plot/cicandmal2017/original_accuracy.csv')
    df_cicandmal2017_adv = pd.read_csv('experiments/plot/cicandmal2017/adversarial_accuracy.csv')

    df_ctu13 = pd.read_csv('experiments/plot/ctu13/original_accuracy.csv')
    df_ctu13_adv = pd.read_csv('experiments/plot/ctu13/adversarial_accuracy.csv')

    df_datacon = pd.read_csv('experiments/plot/datacon/original_accuracy.csv')
    df_datacon_adv = pd.read_csv('experiments/plot/datacon/adversarial_accuracy.csv')

    df_dohbrw = pd.read_csv('experiments/plot/dohbrw/original_accuracy.csv')
    df_dohbrw_adv = pd.read_csv('experiments/plot/dohbrw/adversarial_accuracy.csv')

    n = df_cicandmal2017['algorithm'].unique().__len__()

    # colors = [plt.cm.RdBu_r(i/float(n*5)) for i in range(n)]
    # colors2 = [plt.cm.RdBu(i/float(n*4)) for i in range(n)]

    x = [1]
    for i in range(n-1):
        x.append(x[i]+1.5)

    plt.figure(figsize=(35,6), dpi= 80)

    plt.bar(x=[i - 0.45 for i in x],height=df_cicandmal2017['recall'], color='#ab7332', width=.25, label='CICAndMal2017',alpha=0.7)
    plt.bar(x=[i - 0.45 for i in x],height=df_cicandmal2017_adv['recall'], color='#ab7332', width=.25, label='CICAndMal2017(M-TBA)',alpha=1)

    plt.bar(x=[i - 0.15 for i in x], height=df_ctu13['recall'], color='#2f2216', width=.25, label='CTU-13', alpha=0.6)
    plt.bar(x=[i - 0.15 for i in x], height=df_ctu13_adv['recall'], color='#2f2216', width=.25, label='CTU-13(M-TBA)', alpha=1)

    plt.bar(x=[i + 0.15 for i in x], height=df_datacon['recall'], color='#935c3a', width=.25, label='Datacon2020-EMT', alpha=0.7)
    plt.bar(x=[i + 0.15 for i in x], height=df_datacon_adv['recall'], color='#935c3a', width=.25, label='Datacon2020-EMT(M-TBA)', alpha=1)
    
    plt.bar(x=[i + 0.45 for i in x], height=df_dohbrw['recall'], color='#fec47d', width=.25, label='DoHBrw2020', alpha=0.7)
    plt.bar(x=[i + 0.45 for i in x], height=df_dohbrw_adv['recall'], color='#e6752e', width=.25, label='DoHBrw2020(M-TBA)', alpha=1)

    plt.yticks(fontsize=20,rotation=0, horizontalalignment= 'right',)
    
    
    plt.gca().set_xticks(x)
    plt.gca().set_xticklabels(df_cicandmal2017['algorithm'] , rotation=0, horizontalalignment= 'center',fontdict={'size':20})
    # plt.title("adv_train vs orig", fontsize=14)
    plt.legend(loc='upper center',ncol=4, borderaxespad=-4,prop={'size':13})
    plt.gca().set_ylabel('Adversarial Detection Rate',fontdict={'size':22})
    plt.gca().set_xlabel('Machine Learning Model',fontdict={'size':22}) 
    plt.show()
    # plt.savefig('experiments/figure/datasets_evasion1.pdf', format='pdf', dpi=1000,transparent=True)

def plot_A_TBA():
    df_cicflowmeter = pd.read_csv('experiments/plot/others/cicandmal2017_AM-TBA.csv')
    df_datacon = pd.read_csv('experiments/plot/others/datacon_AM-TBA.csv')
    x_label = [1.2, 2.8, 4.3, 5.75, 7.25, 8.8, 10.35, 11.8]
    n = df_cicflowmeter['attack'].unique().__len__()
    x = [1]
    for i in range(n-1):
        x.append(x[i]+1.5)

    plt.figure(figsize=(22,5))

   # plt.subplots_adjust(left=0.05, bottom=None, right=0.95, top=None,
   #                 wspace=0.1, hspace=None)

    # plt.subplot(121)
    # plt.bar(x=x,height=df_cicflowmeter['M-TBA'], color='#fda56e', width=.4, label='M-TBA',alpha=0.8)
    # plt.bar(x=[i + 0.5 for i in x], height=df_cicflowmeter['AM-TBA'], color='#935c3a', width=.4, label='AM-TBA', alpha=0.8)
    # plt.xticks(x_label)
    # plt.yticks(fontsize=14, rotation=0, horizontalalignment='right', )
    # plt.legend(loc='upper center', ncol=4, borderaxespad=-2, prop={'size': 22})
    # #plt.title("CICAndMAl2017", fontsize=16)
    #
    # plt.gca().set_ylabel('Adversarial Detection Rate', fontdict={'size': 22})
    # plt.gca().set_xlabel('Machine Learning Model', fontdict={'size': 22})
    # plt.gca().set_xticklabels(["LR", "SVM", "MLP", "DT", "Hard-V", "Soft-V", "Stacking", "BayesEns"], rotation=0, fontdict={'size': 19})

    # plt.subplot(122)
    plt.bar(x=x,height=df_datacon['M-TBA'], color='#fda56e', width=.4, label='M-TBA',alpha=0.8)
    plt.bar(x=[i + 0.5 for i in x], height=df_datacon['AM-TBA'], color='#935c3a', width=.4, label='AM-TBA', alpha=0.8)
    plt.xticks(x_label)
    plt.yticks(fontsize=14,rotation=0, horizontalalignment= 'right')
    plt.gca().set_xticklabels(df_cicflowmeter['attack'] , rotation=60, horizontalalignment= 'right')
    #plt.title("Datacon", fontsize=16)

    plt.legend(loc='upper center', ncol=4, borderaxespad=-2, prop={'size': 22})
    plt.gca().set_ylabel('Adversarial Detection Rate', fontdict={'size': 22})
    plt.gca().set_xlabel('Machine Learning Model', fontdict={'size': 22})
    plt.gca().set_xticklabels(["LR", "SVM", "MLP", "DT", "Hard-V", "Soft-V", " Stacking", "BayesEns"], rotation=0, horizontalalignment='center',
                              fontdict={'size': 19})
    plt.show()
    #plt.savefig('experiments/figure/AM-TBA.pdf', format='pdf', dpi=1000,transparent=True)

def plot_adv_train():
    df_adv_train = pd.read_csv('experiments/plot/others/adv_train.csv')
    n = df_adv_train['ml'].unique().__len__()
    x = [1]
    for i in range(n-1):
        x.append(x[i]+1.5)

    plt.figure(figsize=(40,6), dpi= 80)
    # plt.subplots_adjust(left=0.05, bottom=None, right=0.95, top=None,
    #                 wspace=0.05, hspace=None)

    # plt.subplot(111)
    plt.bar(x=x,height=df_adv_train['orig'], color='#fda56e', width=.25, label='ADR(Original)',alpha=1)
    plt.bar(x=[i + 0.3 for i in x], height=df_adv_train['evasion'], color='#180f0a', width=.25, label='ADR(M-TBA)', alpha=1)
    plt.bar(x=[i + 0.6 for i in x], height=df_adv_train['adv_train'], color='#935c3a', width=.25, label='ADR(Adversarial Training)', alpha=1)
    plt.bar(x=[i + 0.9 for i in x], height=df_adv_train['adv_train_orig'], color='#fec47d', width=.25, label='ODR(Adversarial Training)', alpha=1)

    plt.yticks(fontsize=20,rotation=0, horizontalalignment= 'right',)
    
    x_label = [1.7, 3.35, 4.8, 6.15, 7.65, 9.35, 11.05, 12.25, 13.75, 15.35, 16.85, 18.5, 19.95, 21.6, 23.2]
    print(df_adv_train['ml'] )
    plt.gca().set_xticks(x_label)
    plt.gca().set_xticklabels(df_adv_train['ml'] , rotation=0, horizontalalignment= 'right',fontdict={'size':19})
    # plt.title("adv_train vs orig", fontsize=14)
    plt.legend(loc='upper center',ncol=4, borderaxespad=-2,prop={'size':22})
    plt.gca().set_ylabel('Adversarial Detection Rate',fontdict={'size':22})
    plt.gca().set_xlabel('Machine Learning Model',fontdict={'size':22}) 
    plt.show()
    # plt.savefig('../experiments/figure/adv_train_orig.pdf', format='pdf', dpi=1000,transparent=True)

if __name__=='__main__':
    # plot_A_TBA()
    # plot_adv_train()
    plot_feature_evasion()
    # plot_dataset_evasion()