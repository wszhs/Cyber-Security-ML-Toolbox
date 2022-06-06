'''
Author: your name
Date: 2021-05-14 20:18:25
LastEditTime: 2021-05-20 22:49:01
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /Cyber-Security-ML-Toolbox/tests/test_experiments_plot.py
'''
from audioop import bias
import sys
from numpy.core.fromnumeric import size
import pandas as pd
import matplotlib
from sympy import bottom_up
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
# matplotlib.style.use('seaborn-dark')
import numpy as np
from matplotlib import rcParams
config={
    "font.family":'Times New Roman'
}
rcParams.update(config)

def plot_different_datasets():
    # hens deepens
    sheet_name='hens'
    df=pd.read_excel('/Users/zhanghangsheng/Desktop/实验数据/TDSC-CARE/不同攻击数据集对比.xlsx',sheet_name=sheet_name)
    x=[1,2,3,4]
    x_label = [1.5, 2.5, 3.5, 4.5]
    color=['#fdcd78','#fda56e','#a16742','#55321d','#000000']
    # color=['#d8d9dc','#f2e6dc','#dfbac8','#a784a7','#454551']
    # color=['#85bbef','#7084ee','#8970ef','#df81e1','#df39cb']
    # color=['#d8d9dc','#bcd6e6','#5e98c2','#4272b1','#285085']
    color=['#f3cab3','#df9478','#c15a4c','#9e282f','#6e0e20','#5e0000']

    hatch=['','','','','','','']
    if sheet_name=='deepens':
        name=['FGSM','C&W','JSMA','ZOSGD','Bound','HSJA']
    if sheet_name=='hens':
        name=['ZOO','NES','ZOSGD','ZOAda','Bound','HSJA']
    plt.figure(figsize=(15,5), dpi= 80)
    plt.bar(x=[i + 0.15 for i in x], height=df[name[0]]*100, hatch=hatch[0],color=color[0], width=.13, label=name[0], alpha=1)
    plt.bar(x=[i + 0.29 for i in x], height=df[name[1]]*100, color=color[1],hatch=hatch[1], width=.13, label=name[1], alpha=1)
    plt.bar(x=[i + 0.43 for i in x], height=df[name[2]]*100, color=color[2],hatch=hatch[2], width=.13, label=name[2], alpha=1)
    plt.bar(x=[i + 0.57 for i in x], height=df[name[3]]*100, color=color[3],hatch=hatch[3], width=.13, label=name[3], alpha=1)
    plt.bar(x=[i + 0.71 for i in x], height=df[name[4]]*100, color=color[4],hatch=hatch[4],width=.13, label=name[4], alpha=1)
    plt.bar(x=[i + 0.85 for i in x], height=df[name[5]]*100, color=color[5],hatch=hatch[5],width=.13, label=name[5], alpha=1)

    plt.yticks(fontsize=35,rotation=0, horizontalalignment= 'right',)
    plt.gca().set_xticks(x_label)
    plt.gca().set_xticklabels(df['datasets'] , rotation=0, horizontalalignment= 'center',fontdict={'size':25})
    plt.legend(loc='upper center',ncol=6,bbox_to_anchor=(0.5,1.18),prop={'size':20},edgecolor='black')
    plt.gca().set_ylabel('Attack Success Rate (%)',fontdict={'size':30})
    # plt.gca().set_xlabel('',fontdict={'size':28}) 
    plt.savefig('/Users/zhanghangsheng/Desktop/TDSC/NashAE/images/experiment/datasets_'+sheet_name+'.pdf',bbox_inches='tight', format='pdf', dpi=1000,transparent=True)
    # plt.show()
    # print(df)

def plot_AT():
    # bodmas spam ids17 mal17
    sheet_name='ids17'
    df=pd.read_excel('/Users/zhanghangsheng/Desktop/实验数据/TDSC-CARE/对抗训练效果对比.xlsx',sheet_name=sheet_name)
    print(df)
    x=[1,2,3,4,5]
    x_label = [1.5, 2.5, 3.5, 4.5,5.5]

    color=['#f3cab3','#5e0000','#c15a4c','#9e282f']
    color=['#d8d9dc','#285085','#5e98c2','#4272b1','#285085']
    hatch=['','','','','']
    name=['DSR (Original)','DSR (Attack)','DSR (AT)','ODR (AT)']
    bias=[0.2,0.4,0.6,0.8]
    plt.figure(figsize=(15,3.5), dpi= 80)
    for k in range(len(name)):
        plt.bar(x=[i + bias[k] for i in x], height=df[name[k]]*100, hatch=hatch[k],color=color[k], width=.19, label=name[k], alpha=1)
    plt.yticks(fontsize=35,rotation=0, horizontalalignment= 'right',)
    plt.gca().set_xticks(x_label)
    plt.gca().set_xticklabels(df['Model'] , rotation=0, horizontalalignment= 'center',fontdict={'size':35})
    plt.legend(loc='upper center',ncol=6,bbox_to_anchor=(0.5,1.3),prop={'size':20},edgecolor='black')
    # plt.gca().set_ylabel('Defense Success Rate (%)',fontdict={'size':25})
    # plt.savefig('/Users/zhanghangsheng/Desktop/TDSC/NashAE/images/experiment/at_'+sheet_name+'.pdf',bbox_inches='tight', format='pdf', dpi=1000,transparent=True)
    plt.show()

def plot_weight():
    # ids17-attack ids17-defense
    sheet_name='spam-defense'
    df=pd.read_excel('/Users/zhanghangsheng/Desktop/实验数据/TDSC-CARE/权重变化.xlsx',sheet_name=sheet_name)
    color=['#f3cab3','#5e0000','#c15a4c','#9e282f']
    color=['#d8d9dc','#f2e6dc','#dfbac8','#a784a7','#454551']
    # color=['#85bbef','#7084ee','#8970ef','#df81e1','#df39cb']
    color=['#d8d9dc','#bcd6e6','#5e98c2','#4272b1','#285085']
    # color=['#f3cab3','#df9478','#c15a4c','#9e282f','#6e0e20','#5e0000']
    name=['FGSM','JSMA','ZOO','NES']
    bottom_np=df.iloc[:,:-2].values
    K=len(bottom_np[:,0])
    L=len(bottom_np[0])
    bottom_np[:,0]=np.zeros(K)
    x=np.arange(K)+1
    bottom_sum=bottom_np
    for i in range(1,L):
        bottom_sum[:,i]=bottom_sum[:,i-1]+bottom_np[:,i]

    plt.figure(figsize=(30,8), dpi= 80)
    p_arr=[]
    for j in range(L):
        p=plt.bar(x,df[name[j]],width=.7,bottom=bottom_sum[:,j],color=color[j])
        p_arr.append(p)
    plt.plot(x,df['Score'],marker="o",markerfacecolor='Red',linewidth=2,markersize=20,color='#9e282f')
    count=0
    for a, b in zip(x, df['Score']):
        count+=1
        if count%5==0:
            if count==5:
                plt.text(a, b, 'DSR_all='+format(b*100, '.2f')+"%", ha='center', va='bottom',color='Black', fontsize=35)
            else:
                plt.text(a, b, format(b*100, '.2f')+"%", ha='center', va='bottom',color='Black', fontsize=40)
    plt.xticks(fontsize=45)
    plt.yticks(fontsize=45,rotation=0, horizontalalignment= 'right',)
    # plt.legend(p_arr,name,loc='upper center',ncol=6,bbox_to_anchor=(0.5,1.15),prop={'size':25},edgecolor='black')
    plt.gca().set_ylabel('Defense Weight Vectors',fontdict={'size':45})
    # plt.show()
    plt.savefig('/Users/zhanghangsheng/Desktop/TDSC/NashAE/images/experiment/weight_'+sheet_name+'.pdf',bbox_inches='tight', format='pdf', dpi=1000,transparent=True)

def plot_eps():
    # spam ids17
    sheet_name='ids17'
    df = pd.read_excel('/Users/zhanghangsheng/Desktop/实验数据/TDSC-CARE/不同的eps.xlsx',sheet_name=sheet_name)
    K=5
    name=['MLP','Xgboost','DeepEns','TreeEns','HeteroEns']
    color=['#f1595e','#ff1887','#8971e1','#007417','#56cefc']
    marker=['o','x','*','^','s']
    x=np.arange(len(df[name[0]]))+1
    plt.figure(figsize=(10,6), dpi= 80)
    p_arr=[]
    for i in range(K):
        p=plt.plot(x,df[name[i]],marker=marker[i],markerfacecolor='White',linewidth=2,markersize=20,color=color[i],label=name[i])
        p_arr.append(p)
    # plt.lgend(loc='upper center',ncol=5,bbox_to_anchor=(0.5,1.2),prop={'size':35},fancybox=True,shadow=False)
    plt.xticks(fontsize=35)
    plt.yticks(fontsize=35,rotation=0, horizontalalignment= 'right',)
    plt.gca().set_xticklabels(df['eps-x'] , rotation=0, horizontalalignment= 'center',fontdict={'size':35})
    plt.gca().set_ylabel('Attack Success Rate (%)',fontdict={'size':35})
    plt.grid()
    # plt.show()
    plt.savefig('/Users/zhanghangsheng/Desktop/TDSC/NashAE/images/experiment/eps_'+sheet_name+'.pdf',bbox_inches='tight', format='pdf', dpi=1000,transparent=True)
    # plt.savefig('/Users/zhanghangsheng/Desktop/TDSC/NashAE/images/experiment/leg_eps'+sheet_name+'.pdf',bbox_inches='tight', format='pdf', dpi=1000,transparent=True)
    
def plot_transfer():
    # bodmas ids17 spam mal17
    sheet_name='mal17'
    df = pd.read_excel('/Users/zhanghangsheng/Desktop/实验数据/TDSC-CARE/不同的攻击方法与对抗训练.xlsx',sheet_name=sheet_name)
    # print(df)
    plt.figure(figsize=(15,8.5), dpi= 80)
    y_label=df['Attack'].values
    df=df.drop(['Attack'], axis=1)
    h = sns.heatmap((100-df),vmin=0,vmax=100, xticklabels = df.columns, yticklabels = y_label, cmap='RdBu_r',
                        annot=True,fmt='.2f',annot_kws={'size':25},cbar=False)

    cb=h.figure.colorbar(h.collections[0]) #显示colorbar
    cb.ax.tick_params(labelsize=35) #设置colorbar刻度字体大小。

    plt.xticks(fontsize=25)
    plt.yticks(fontsize=30,rotation=0, horizontalalignment= 'right')
    plt.yticks([])
    plt.savefig('/Users/zhanghangsheng/Desktop/TDSC/NashAE/images/experiment/transfer_'+sheet_name+'.pdf',bbox_inches='tight', format='pdf', dpi=1000,transparent=True)
    # plt.show()

def plot_ada_ens():
    # ids17-ens ids17-trans spam-ens spam-trans
    sheet_name='ids17-trans'
    df=pd.read_excel('/Users/zhanghangsheng/Desktop/实验数据/TDSC-CARE/自适应集成攻击.xlsx',sheet_name=sheet_name)
    print(df)
    x=[1,2,3,4,5]
    x_label = [1.5, 2.5, 3.5, 4.5,5.5]
    # color=['#d8d9dc','#bcd6e6','#5e98c2','#4272b1','#285085']
    color=['#f3cab3','#df9478','#c15a4c','#9e282f','#6e0e20','#5e0000']
    hatch=['','','','','']
    if 'ens' in sheet_name:
        name=['ZOO','ZOSGD','Avg-MA','Adp-MA']
    else:
        name=['MLP','Xgb','Avg-MA','Adp-MA']
    bias=[0.2,0.4,0.6,0.8]
    plt.figure(figsize=(15,4), dpi= 80)
    for k in range(len(name)):
        plt.bar(x=[i + bias[k] for i in x], height=df[name[k]]*100, hatch=hatch[k],color=color[k], width=.19, label=name[k], alpha=1)
    plt.yticks(fontsize=30,rotation=0, horizontalalignment= 'right',)
    plt.gca().set_xticks(x_label)
    plt.gca().set_xticklabels(df['Model'] , rotation=0, horizontalalignment= 'center',fontdict={'size':35})
    # plt.legend(loc='upper center',ncol=6,bbox_to_anchor=(0.5,1.3),prop={'size':25},edgecolor='black')
    plt.gca().set_ylabel('Attack Success Rate (%)',fontdict={'size':25})
    plt.savefig('/Users/zhanghangsheng/Desktop/TDSC/NashAE/images/experiment/ada_'+sheet_name+'.pdf',bbox_inches='tight', format='pdf', dpi=1000,transparent=True)
    # plt.show()

def plot_transfer_AT():
    # ids17-mlp ids17-max
    sheet_name='ids17-max'
    df=pd.read_excel('/Users/zhanghangsheng/Desktop/实验数据/TDSC-CARE/迁移对抗训练.xlsx',sheet_name=sheet_name)
    x=np.arange(len(df['DSR']))+1
    name=['MLP','Xgboost','DeepEns','HeteroEns']
    color=['#f1595e','#ff1887','#8971e1','#007417','#56cefc']
    marker=['o','x','*','^','s']
    K=4
    fig, ax_arr = plt.subplots(2, 2, sharex=True, figsize=(10, 5))
    x_label = [1, 2, 3, 4,5,6,7,8]

    for k in range(K):
        row=int((k)/2)
        col=k%2
        ax_arr[row,col].plot(x,df['DSR'],marker=marker[0],markerfacecolor='White',linewidth=2,markersize=10,color=color[1],label='Attack')
        ax_arr[row,col].plot(x,df[name[k]],marker=marker[1],markerfacecolor='White',linewidth=2,markersize=10,color='Blue',label='Training')
        ax_arr[row,col].fill_between(x,df['DSR'],df[name[k]],alpha=0.2)
        ax_arr[row,col].set_title(name[k],fontsize=15)
        # set_ylabel('Defense Success Rate (%)',fontdict={'size':25})
        ax_arr[row,col].set_xticks(x_label)
        ax_arr[row,col].set_xticklabels(df['Attack'] , rotation=0, horizontalalignment= 'center',fontdict={'size':15})
        ax_arr[row,col].set_yticklabels([-20,0,20,40,60,80,100],fontdict={'size':20})
    plt.yticks(fontsize=15,rotation=0, horizontalalignment= 'right',)
    plt.legend(loc='upper center',ncol=6,bbox_to_anchor=(-0.1,2.55),prop={'size':15},edgecolor='black')
    plt.gca().set_ylabel('Defense Success Rate (%)',fontdict={'size':15})
    plt.show()
    # plt.savefig('/Users/zhanghangsheng/Desktop/TDSC/NashAE/images/experiment/transfer_at_'+sheet_name+'.pdf',bbox_inches='tight', format='pdf', dpi=1000,transparent=True)

def plot_problem_attack():
    sheet_name='black'
    marker=['o','x','*','^','s']
    color=['#f1595e','#ff1887','#8971e1','#007417','#56cefc']
    df=pd.read_excel('/Users/zhanghangsheng/Desktop/实验数据/TDSC-CARE/问题空间攻击.xlsx',sheet_name=sheet_name)
    print(df)
    x=np.arange(len(df['OSR']))+1
    x_label = [1, 2, 3, 4,5,6]

    plt.plot(x,df['OSR'],marker=marker[0],markerfacecolor='White',linewidth=2,markersize=10,color='Blue')
    plt.fill_between(x,df['OSR'],df['ASR'],alpha=0.2)
    plt.plot(x,df['ASR'],marker=marker[0],markerfacecolor='White',linewidth=2,markersize=10,color=color[1])
    for a, b in zip(x, df['ASR']):
        plt.text(a,b,b, ha='center', va='bottom',color='Black', fontsize=20)
    plt.gca().set_xticks(x_label)
    plt.gca().set_xticklabels(df['Attack'] , rotation=0, horizontalalignment= 'center',fontdict={'size':15})
    plt.gca().set_ylabel('Detection Rate',fontdict={'size':25})
    plt.yticks(fontsize=25,rotation=0, horizontalalignment= 'right',)
    plt.savefig('/Users/zhanghangsheng/Desktop/TDSC/NashAE/images/experiment/pro_'+sheet_name+'.pdf',bbox_inches='tight', format='pdf', dpi=1000,transparent=True)
    # plt.show()



if __name__=='__main__':

    # plot_different_datasets()
    # plot_eps()
    # plot_AT()
    # plot_ada_ens()
    # plot_weight()
    plot_transfer()
    # plot_transfer_AT()
    # plot_problem_attack()


