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
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
# matplotlib.style.use('seaborn-whitegrid')
import numpy as np
from matplotlib import rcParams
config={
    "font.family":'Times New Roman'
}
rcParams.update(config)


def plot_different_model():
    # df=pd.read_excel('/Users/zhanghangsheng/Desktop/实验数据/TIFS实验/1.2-不同攻击数据集对比.xlsx',)
    df = pd.read_excel('experiments/plot/kitsune/different.xlsx',sheet_name='Sheet1')
    x=[1,2,3,4,5,6]
    x_label = [1.5, 2.5, 3.5, 4.5, 5.5,6.5]
    # color=['#fdcd78','#fda56e','#a16742','#55321d','#000000']
    # color=['#d8d9dc','#f2e6dc','#dfbac8','#a784a7','#454551']
    # color=['#f1595e','#ff1887','#8971e1','#007417','#56cefc']
    color=['#85bbef','#7084ee','#8970ef','#df81e1','#df39cb']
    hatch=['','','','','']
    plt.figure(figsize=(20,3), dpi= 80)
    plt.bar(x=[i + 0.2 for i in x], height=1-df['E-JSMA'], hatch=hatch[0],color=color[0], width=.14, label='E-JSMA', alpha=1)
    plt.bar(x=[i + 0.35 for i in x], height=1-df['T-CW'], color=color[1],hatch=hatch[1], width=.14, label='T-CW', alpha=1)
    plt.bar(x=[i + 0.5 for i in x], height=1-df['B-M'], color=color[2],hatch=hatch[2], width=.14, label='B-M', alpha=1)
    plt.bar(x=[i + 0.65 for i in x], height=1-df['SI'], color=color[3],hatch=hatch[3], width=.14, label='SI', alpha=1)
    plt.bar(x=[i + 0.8 for i in x], height=1-df['Ours'], color=color[4],hatch=hatch[4],width=.14, label='Ours', alpha=1)

    plt.yticks(fontsize=25,rotation=0, horizontalalignment= 'right',)
    plt.gca().set_xticks(x_label)
    plt.gca().set_xticklabels(df['attack'] , rotation=0, horizontalalignment= 'center',fontdict={'size':23})
    plt.legend(loc='upper center',ncol=5,bbox_to_anchor=(0.5,1.3),prop={'size':20},fancybox=True,shadow=True)
    # plt.legend(loc='upper center',ncol=4, borderaxespad=-4,prop={'size':13})
    # plt.gca().set_ylabel('DER',fontdict={'size':28})
    # plt.gca().set_xlabel('',fontdict={'size':28}) 
    # plt.savefig('experiments/figure/zhs2/different_kitsune_model.pdf',bbox_inches='tight', format='pdf', dpi=1000,transparent=True)
    plt.show()
    # print(df)

def plot_afterimage_model():
    df = pd.read_excel('experiments/plot/kitsune/botnet.xlsx',sheet_name='Sheet1')
    x=[1,2,3,4,5,6,7,8,9,10,11]
    x_label = [1.5, 2.5, 3.5, 4.5, 5.5,6.5,7.5,8.5,9.5,10.5,11.5]
    # color=['#fdcd78','#fda56e','#a16742','#55321d','#000000']
    color=['#d8d9dc','#f2e6dc','#dfbac8','#a784a7','#454551']
    hatch=['','','','','']
    plt.figure(figsize=(20,3), dpi= 80)
    plt.bar(x=[i + 0.2 for i in x], height=1-df['lr+sgd'], hatch=hatch[0],color=color[0], width=.14, label='LR-SGD', alpha=1)
    plt.bar(x=[i + 0.35 for i in x], height=1-df['lr+f'], color=color[1],hatch=hatch[1], width=.14, label='LR-F', alpha=1)
    plt.bar(x=[i + 0.5 for i in x], height=1-df['ensemble+f'], color=color[2],hatch=hatch[2], width=.14, label='Ens-F', alpha=1)
    plt.bar(x=[i + 0.65 for i in x], height=1-df['min-max+f'], color=color[3],hatch=hatch[3], width=.14, label='Min-Max-F', alpha=1)
    plt.bar(x=[i + 0.8 for i in x], height=1-df['min-max+fi'], color=color[4],hatch=hatch[4],width=.14, label='Min-Max-FI', alpha=1)

    plt.yticks(fontsize=25,rotation=0, horizontalalignment= 'right',)
    plt.gca().set_xticks(x_label)
    plt.gca().set_xticklabels(df['algorithm'] , rotation=0, horizontalalignment= 'center',fontdict={'size':23})
    plt.legend(loc='upper center',ncol=5,bbox_to_anchor=(0.5,1.3),prop={'size':20},fancybox=True,shadow=True)
    # plt.legend(loc='upper center',ncol=4, borderaxespad=-4,prop={'size':13})
    # plt.gca().set_ylabel('DER',fontdict={'size':28})
    # plt.gca().set_xlabel('',fontdict={'size':28}) 
    # plt.savefig('experiments/figure/zhs/afterimage_model.pdf',bbox_inches='tight', format='pdf', dpi=1000,transparent=True)
    plt.show()
    # print(df)

def plot_cicflowmeter_model():
    df = pd.read_excel('experiments/plot/cicids2017/botnet.xlsx',sheet_name='Sheet1')
    x=[1,2,3,4,5,6,7,8,9,10,11]
    x_label = [1.5, 2.5, 3.5, 4.5, 5.5,6.5,7.5,8.5,9.5,10.5,11.5]
    # color=['#fdcd78','#fda56e','#a16742','#55321d','#000000']
    color=['#d8d9dc','#f2e6dc','#dfbac8','#a784a7','#454551']
    hatch=['','','','','']
    plt.figure(figsize=(20,3), dpi= 80)
    plt.bar(x=[i + 0.2 for i in x], height=1-df['lr+sgd'], hatch=hatch[0],color=color[0], width=.14, label='LR-SGD', alpha=1)
    plt.bar(x=[i + 0.35 for i in x], height=1-df['lr+f'], color=color[1],hatch=hatch[1], width=.14, label='LR-F', alpha=1)
    plt.bar(x=[i + 0.5 for i in x], height=1-df['ensemble+f'], color=color[2],hatch=hatch[2], width=.14, label='Ens-F', alpha=1)
    plt.bar(x=[i + 0.65 for i in x], height=1-df['min-max+f'], color=color[3],hatch=hatch[3], width=.14, label='Min-Max-F', alpha=1)
    plt.bar(x=[i + 0.8 for i in x], height=1-df['min-max+fi'], color=color[4],hatch=hatch[4],width=.14, label='Min-Max-FI', alpha=1)

    plt.yticks(fontsize=25,rotation=0, horizontalalignment= 'right',)
    plt.gca().set_xticks(x_label)
    plt.gca().set_xticklabels(df['algorithm'] , rotation=0, horizontalalignment= 'center',fontdict={'size':23})
    plt.legend(loc='upper center',ncol=5,bbox_to_anchor=(0.5,1.3),prop={'size':20},fancybox=True,shadow=True)
    # plt.legend(loc='upper center',ncol=4, borderaxespad=-4,prop={'size':13})
    # plt.gca().set_ylabel('DER',fontdict={'size':28})
    # plt.gca().set_xlabel('',fontdict={'size':28}) 
    plt.savefig('experiments/figure/zhs/cicids2017_model.pdf',bbox_inches='tight', format='pdf', dpi=1000,transparent=True)
    # plt.show()
    # print(df)

def plot_feature_evasion():
    df = pd.read_excel('experiments/plot/kitsune/feature.xlsx',sheet_name='Sheet1')
    x=[1,2,3,4,5,6,7,8,9,10,11]
    x_label = [1.5, 2.5, 3.5, 4.5, 5.5,6.5,7.5,8.5,9.5,10.5,11.5]
    # color=['#000000','#55321d','#a16742','#ffcd78']
    color=['#454551','#dfbac8','#a784a7','#4d6572']
    hatch=['','','','','']
    plt.figure(figsize=(20,3), dpi= 80)
    # df['kitsune-botnet-a'].plot(kind='line')
    plt.bar(x=[i + 0.2 for i in x], height=df['kitsune-botnet-a'], hatch=hatch[0],color=color[0], width=.19, label='kitsune-botnet-a', alpha=1)
    plt.bar(x=[i + 0.2 for i in x], height=df['kitsune-botnet-r'], hatch=hatch[0],color=color[0], width=.19, label='kitsune-botnet-r', alpha=0.3)

    plt.bar(x=[i + 0.4 for i in x], height=df['cic-botnet-a'], color=color[1],hatch=hatch[1], width=.19, label='cicids-botnet-a', alpha=1)
    plt.bar(x=[i + 0.4 for i in x], height=df['cic-botnet-r'], color=color[1],hatch=hatch[1], width=.19, label='cicids-botnet-r', alpha=0.6)

    plt.bar(x=[i + 0.6 for i in x], height=df['kitsune-ddos-a'], color=color[2],hatch=hatch[2], width=.19, label='kitsune-ddos-a', alpha=1)
    plt.bar(x=[i + 0.6 for i in x], height=df['kitsune-ddos-r'], color=color[2],hatch=hatch[2], width=.19, label='kitsune-ddos-r', alpha=0.6)

    plt.bar(x=[i + 0.8 for i in x], height=df['cic-ddos-a'], color=color[3],hatch=hatch[3], width=.19, label='cicids-ddos-a', alpha=1)
    plt.bar(x=[i + 0.8 for i in x], height=df['cic-ddos-r'], color=color[3],hatch=hatch[3], width=.19, label='cicids-ddos-r', alpha=0.6)

    plt.yticks(fontsize=25,rotation=0, horizontalalignment= 'right',)
    plt.gca().set_xticks(x_label)
    plt.gca().set_xticklabels(df['algorithm'] , rotation=0, horizontalalignment= 'center',fontdict={'size':23})
    plt.legend(loc='upper center',ncol=4,bbox_to_anchor=(0.5,1.5),prop={'size':20},fancybox=True,shadow=True)
    # plt.savefig('experiments/figure/zhs/feature.pdf',bbox_inches='tight', format='pdf', dpi=1000,transparent=True)
    plt.show()
    # print(df)

def plot_transfer():
    my_color=['#89f9e8','#7ee9ee','#51a4f2','#4275f4','#5753f4','#913af3','#963af5','#b937f2','#d739d5','#d538c6']
    # my_color=['#161d15','#53565c','#75809c','#f2e6dc','#dfbac8','#a784a7','#965454']
    df = pd.read_excel('experiments/plot/kitsune/botnet_transfer.xlsx',sheet_name='Sheet1')
    # df = pd.read_excel('experiments/plot/kitsune/ddos_transfer.xlsx',sheet_name='Sheet1')
    # df = pd.read_excel('experiments/plot/cicids2017/botnet_transfer.xlsx',sheet_name='Sheet1')
    # df = pd.read_excel('experiments/plot/cicids2017/ddos_transfer.xlsx',sheet_name='Sheet1')
    plt.figure(figsize=(21,9), dpi= 80)
    y_label=df['attack'].values
    df=df.drop(['attack'], axis=1)
    h = sns.heatmap(1-df,vmin=0,vmax=1, xticklabels = df.columns, yticklabels = y_label, cmap=my_color,
                        annot=True,annot_kws={'size':22},cbar=False)

    cb=h.figure.colorbar(h.collections[0]) #显示colorbar
    cb.ax.tick_params(labelsize=26) #设置colorbar刻度字体大小。

    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22,rotation=0, horizontalalignment= 'right')
    # plt.show()
    plt.savefig('experiments/figure/zhs2/kitsune_botnet_transfer.pdf',bbox_inches='tight', format='pdf', dpi=1000,transparent=True)
    # plt.savefig('experiments/figure/zhs2/kitsune_ddos_transfer.pdf',bbox_inches='tight', format='pdf', dpi=1000,transparent=True)
    # plt.savefig('experiments/figure/zhs2/cicids2017_botnet_transfer.pdf',bbox_inches='tight', format='pdf', dpi=1000,transparent=True)
    # plt.savefig('experiments/figure/zhs2/cicids2017_ddos_transfer.pdf',bbox_inches='tight', format='pdf', dpi=1000,transparent=True)
    
def plot_line():
    matplotlib.style.use('ggplot')
    df = pd.read_excel('experiments/plot/kitsune/inter.xlsx',sheet_name='Sheet1')
    print(df)
    x=[1,2,3,4,5]
    x_label = [1.5, 2.5, 3.5, 4.5, 5.5,6.5,7.5,8.5,9.5,10.5,11.5]
    color=['#000000','#55321d','#a16742','#ffcd78']
    hatch=['','','','','']
    plt.figure(figsize=(7,3), dpi= 80)
    
    plt.plot(x,1-df['LR'],"co-",linewidth=2,markersize=10,color='#df81e1',label='LR')
    plt.plot(x,1-df['DT'],"r.:",linewidth=2,markersize=10,color='#fda0c5',label='DT')
    plt.plot(x,1-df['SVM'],"gh-.",linewidth=2,markersize=10,color='#8971e1',label='SVM')
    plt.plot(x,1-df['MLP'],"bd:",linewidth=2,markersize=10,color='#06aff1',label='MLP')
    plt.plot(x,1-df['RF'],"c*--",linewidth=2,markersize=10,color='#4573c5',label='RF')
    plt.plot(x,1-df['Ens'],"rp-",linewidth=2,markersize=10,color='#fec103',label='Ens')

    plt.yticks(fontsize=20,rotation=0, horizontalalignment= 'right',)
    plt.gca().set_xticks(x)
    plt.gca().set_xticklabels(x, rotation=0, horizontalalignment= 'center',fontdict={'size':20})
    plt.legend(loc='upper center',ncol=3,bbox_to_anchor=(0.7,0.95),prop={'size':12},fancybox=True,shadow=True)
    plt.show()
    # plt.savefig('experiments/figure/zhs/inter.pdf',bbox_inches='tight', format='pdf', dpi=1000,transparent=False)

def plot_diff_feature():
    matplotlib.style.use('ggplot')
    # df = pd.read_excel('experiments/plot/kitsune/diff-feature.xlsx',sheet_name='Sheet1')
    df = pd.read_excel('experiments/plot/cicids2017/diff-feature.xlsx',sheet_name='Sheet1')
    print(df)
    # color=['#a784a7','#454551']
    color=['#7084ee','#df39cb']
    x=[0.4,1.4,2.4,3.4,4.4]
    x_label = ['LR','LR','MLP', 'Xgb', 'Ens', 'Min-Max']
    df_label = ['lr','mlp', 'xgboost', 'ensemble', 'min-max']
    algorithm=['Kitnet','Diff-rf','IF','LR','DT','SVM','MLP','RF','XGb','DF','Ens']
    plt.figure(figsize=(20,10), dpi= 80)

    # p1=plt.bar(x=x[0]+0.4, height=1-df.iloc[0][df_label[0]+'+sgd'],color=color[0], width=.4, label='SGD', alpha=1)
    # p2=plt.bar(x=x[0]+0.8, height=1-df.iloc[0][df_label[0]+'+f'],color=color[1], width=.4, label='F', alpha=1)
    for j in range(6):
        plt.subplot(23*10+j+1)
        for i in range(0,len(df_label)):
            plt.bar(x=x[i]+0.4, height=1-df.iloc[j][df_label[i]+'+sgd'],color=color[0], width=.4, label='SGD', alpha=0.9)
            plt.bar(x=x[i]+0.8, height=1-df.iloc[j][df_label[i]+'+f'],color=color[1], width=.4, label='F', alpha=0.9)
        plt.title(algorithm[j],fontdict={'size':20}) 
        plt.yticks(fontsize=15,rotation=0, horizontalalignment= 'right',)
        plt.gca().set_xticklabels(x_label,fontdict={'size':15})

    # plt.show()
    # plt.savefig('experiments/figure/zhs2/kitsune-diff.pdf',bbox_inches='tight', format='pdf', dpi=1000,transparent=False)
    plt.savefig('experiments/figure/zhs2/cicids2017-diff.pdf',bbox_inches='tight', format='pdf', dpi=1000,transparent=False)

def plot_del_feature_line():
    # matplotlib.style.use('ggplot')
    df = pd.read_excel('experiments/plot/kitsune/del_robustness.xlsx',sheet_name='Sheet1')
    print(df)
    x=df['dim']
    x_label = [1.5, 2.5, 3.5, 4.5, 5.5,6.5,7.5,8.5,9.5,10.5,11.5]
    color=['#f1595e','#ff1887','#8971e1','#007417','#56cefc']
    plt.figure(figsize=(7,7), dpi= 80)
    
    plt.plot(x,df['LR'],marker="o",markerfacecolor='white',linewidth=2,markersize=10,color=color[0],label='LR')
    plt.plot(x,df['DT'],marker="x",markerfacecolor='white',linewidth=2,markersize=10,color=color[1],label='DT')
    plt.plot(x,df['SVM'],marker='*',markerfacecolor='white',linewidth=2,markersize=10,color=color[2],label='SVM')
    plt.plot(x,df['MLP'],marker="^",markerfacecolor='white',linewidth=2,markersize=10,color=color[3],label='MLP')
    plt.plot(x,df['Ens'],marker="s",markerfacecolor='white',linewidth=2,markersize=10,color=color[4],label='Ens')

    plt.yticks(fontsize=25,rotation=0, horizontalalignment= 'right',)
    plt.gca().set_xticks(x)
    plt.gca().set_xticklabels(x, rotation=0, horizontalalignment= 'center',fontdict={'size':25})
    plt.legend(loc='upper center',ncol=1,bbox_to_anchor=(0.8,0.995),prop={'size':20},fancybox=True,shadow=False)
    plt.xlabel("%Dimensions",fontsize=25)
    plt.ylabel("Attack Success Rate",fontsize=25)
    plt.grid()
    # plt.show()
    plt.savefig('experiments/figure/zhs2/del-feature-line.pdf',bbox_inches='tight', format='pdf', dpi=1000,transparent=False)

def plot_use_feature_line():
    # matplotlib.style.use('ggplot')
    df = pd.read_excel('experiments/plot/kitsune/use_robustness.xlsx',sheet_name='Sheet1')
    print(df)
    x=df['dim']
    x_label = [1.5, 2.5, 3.5, 4.5, 5.5,6.5,7.5,8.5,9.5,10.5,11.5]
    color=['#f1595e','#ff1887','#8971e1','#007417','#56cefc']
    plt.figure(figsize=(7,7), dpi= 80)
    
    plt.plot(x,df['LR'],marker="o",markerfacecolor='white',linewidth=2,markersize=10,color=color[0],label='LR')
    plt.plot(x,df['DT'],marker="x",markerfacecolor='white',linewidth=2,markersize=10,color=color[1],label='DT')
    plt.plot(x,df['SVM'],marker='*',markerfacecolor='white',linewidth=2,markersize=10,color=color[2],label='SVM')
    plt.plot(x,df['MLP'],marker="^",markerfacecolor='white',linewidth=2,markersize=10,color=color[3],label='MLP')
    plt.plot(x,df['Ens'],marker="s",markerfacecolor='white',linewidth=2,markersize=10,color=color[4],label='Ens')

    plt.yticks(fontsize=25,rotation=0, horizontalalignment= 'right',)
    plt.gca().set_xticks(x)
    plt.gca().set_xticklabels(x, rotation=0, horizontalalignment= 'center',fontdict={'size':25})
    plt.legend(loc='upper center',ncol=1,bbox_to_anchor=(0.8,0.995),prop={'size':20},fancybox=True,shadow=False)
    plt.xlabel("%Dimensions",fontsize=25)
    plt.ylabel("Recall",fontsize=25)
    plt.grid()
    # plt.show()
    plt.savefig('experiments/figure/zhs2/use-feature.pdf',bbox_inches='tight', format='pdf', dpi=1000,transparent=False)

def plot_similar_line():
    # matplotlib.style.use('ggplot')
    df = pd.read_excel('experiments/plot/kitsune/similar.xlsx',sheet_name='Sheet1')
    print(df)
    x=df['dim']
    x_label = [1.5, 2.5, 3.5, 4.5, 5.5,6.5,7.5,8.5,9.5,10.5,11.5]
    color=['#f1595e','#ff1887','#8971e1','#007417','#56cefc']
    plt.figure(figsize=(7,7), dpi= 80)
    
    plt.plot(x,df['LR'],marker="o",markerfacecolor='white',linewidth=2,markersize=10,color=color[0],label='LR')
    plt.plot(x,df['DT'],marker="x",markerfacecolor='white',linewidth=2,markersize=10,color=color[1],label='DT')
    plt.plot(x,df['SVM'],marker='*',markerfacecolor='white',linewidth=2,markersize=10,color=color[2],label='SVM')
    plt.plot(x,df['MLP'],marker="^",markerfacecolor='white',linewidth=2,markersize=10,color=color[3],label='MLP')
    plt.plot(x,df['Ens'],marker="s",markerfacecolor='white',linewidth=2,markersize=10,color=color[4],label='Ens')

    plt.yticks(fontsize=25,rotation=0, horizontalalignment= 'right',)
    plt.gca().set_xticks(x)
    plt.gca().set_xticklabels(x, rotation=0, horizontalalignment= 'center',fontdict={'size':25})
    plt.legend(loc='upper center',ncol=1,bbox_to_anchor=(0.8,0.45),prop={'size':20},fancybox=True,shadow=False)
    plt.xlabel("%Similar",fontsize=25)
    plt.ylabel("Transferability",fontsize=25)
    plt.grid()
    # plt.show()
    plt.savefig('experiments/figure/zhs2/similar.pdf',bbox_inches='tight', format='pdf', dpi=1000,transparent=False)

if __name__=='__main__':

    # plot_different_model()
    # plot_feature_evasion()
    # plot_afterimage_model()
    # plot_cicflowmeter_model()
    # plot_transfer()
    # plot_line()
    # plot_diff_feature()

    #new
    plot_different_model()
    # plot_transfer()
    # plot_diff_feature()
    # plot_del_feature_line()
    # plot_use_feature_line()
    # plot_similar_line()

