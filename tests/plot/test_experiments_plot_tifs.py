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
# matplotlib.style.use('seaborn-dark')
import numpy as np
from matplotlib import rcParams
config={
    "font.family":'Times New Roman'
}
rcParams.update(config)

def plot_different_datasets():
    # ids17 kitsune
    sheet_name='ids17'
    df=pd.read_excel('/Users/zhanghangsheng/Desktop/实验数据/TIFS实验/1.2-不同攻击数据集对比2.xlsx',sheet_name=sheet_name)
    x=[1,2,3,4]
    x_label = [1.5, 2.5, 3.5, 4.5]
    # color=['#fdcd78','#fda56e','#a16742','#55321d','#000000']
    # color=['#d8d9dc','#f2e6dc','#dfbac8','#a784a7','#454551']
    # color=['#f1595e','#ff1887','#8971e1','#007417','#56cefc']
    # color=['#85bbef','#7084ee','#8970ef','#df81e1','#df39cb']
    # color=['#d8d9dc','#bcd6e6','#5e98c2','#4272b1','#285085']
    color=['#f3cab3','#df9478','#c15a4c','#9e282f','#5e0e20']
    hatch=['','','','','']
    plt.figure(figsize=(15,5), dpi= 80)
    plt.bar(x=[i + 0.2 for i in x], height=df['JSMA']*100, hatch=hatch[0],color=color[0], width=.14, label='JSMA', alpha=1)
    plt.bar(x=[i + 0.35 for i in x], height=df['CW']*100, color=color[1],hatch=hatch[1], width=.14, label='C&W', alpha=1)
    plt.bar(x=[i + 0.5 for i in x], height=df['ZOO']*100, color=color[2],hatch=hatch[2], width=.14, label='ZOO', alpha=1)
    plt.bar(x=[i + 0.65 for i in x], height=df['HSJA']*100, color=color[3],hatch=hatch[3], width=.14, label='HSJA', alpha=1)
    plt.bar(x=[i + 0.8 for i in x], height=df['Ours']*100, color=color[4],hatch=hatch[4],width=.14, label='Ours', alpha=1)

    plt.yticks(fontsize=35,rotation=0, horizontalalignment= 'right',)
    plt.gca().set_xticks(x_label)
    plt.gca().set_xticklabels(df['attack'] , rotation=0, horizontalalignment= 'center',fontdict={'size':25})
    plt.legend(loc='upper center',ncol=5,bbox_to_anchor=(0.5,1.18),prop={'size':20},edgecolor='black')
    plt.gca().set_ylabel('Average Attack Success Rate (%)',fontdict={'size':22})
    # plt.gca().set_xlabel('',fontdict={'size':28}) 
    plt.savefig('/Users/zhanghangsheng/Desktop/TDSC/ETA/images/experiment/'+'different_'+sheet_name+'.pdf',bbox_inches='tight', format='pdf', dpi=1000,transparent=True)
    # plt.show()

def plot_transfer():
    # kitsune-botnet kitsune-ddos ids17-botnet ids17-ddos
    sheet_name='kitsune-botnet'
    my_color=['#161d15','#53565c','#858b92','#d8d9dc','#bcd6e6','#5e98c2','#4272b1','#285085']
    # my_color=['#161d15','#53565c','#75809c','#f2e6dc','#dfbac8','#a784a7','#965454']
    df = pd.read_excel('/Users/zhanghangsheng/Desktop/实验数据/TIFS实验/2.1-不同基模型的迁移性.xlsx',sheet_name=sheet_name)
    # print(df)
    plt.figure(figsize=(15,10), dpi= 80)
    y_label=df['attack'].values
    df=df.drop(['attack'], axis=1)
    print((1-df)*100)
    h = sns.heatmap((1-df)*100,vmin=0,vmax=100, xticklabels = df.columns, yticklabels = y_label, cmap='RdBu_r',
                        annot=True,fmt='.2f',annot_kws={'size':28},cbar=False)

    # cb=h.figure.colorbar(h.collections[0]) #显示colorbar
    # cb.ax.tick_params(labelsize=40) #设置colorbar刻度字体大小。

    plt.xticks(fontsize=38)
    plt.yticks(fontsize=40,rotation=0, horizontalalignment= 'right')
    plt.yticks([])
    # plt.show()
    plt.savefig('/Users/zhanghangsheng/Desktop/TDSC/ETA/images/experiment/'+sheet_name+'_transfer.pdf',bbox_inches='tight', format='pdf', dpi=1000,transparent=True)
    
def plot_diff_feature():
    # matplotlib.style.use('ggplot')
    # ids17 kitsune
    sheet_name='ids17'
    df = pd.read_excel('/Users/zhanghangsheng/Desktop/实验数据/TIFS实验/2.2-是否使用重要特征.xlsx',sheet_name=sheet_name)
    color=['#4272b1','#285085']
    x=[-0.2,0.8,1.8,2.8,3.8,4.8,5.8,6.8]
    x_label = ['LR','LR','DT','SVM','MLP','XGb','Ens','Kitnet','Diff']
    substitute=['LR','XGb','Ens','Min-Max']
    df_label=['lr','xgboost','ensemble','min-max']
    plt.figure(figsize=(25,10), dpi= 80)
    
    for j in range(4):
        plt.subplot(22*10+j+1)
        plt.bar([i for i in x],height=(1-df[df_label[j]+'+sgd'])*100,color=color[0],width=.4,label='Non-ISFS',alpha=0.6)
        plt.bar([i + 0.4 for i in x],height=(1-df[df_label[j]+'+f'])*100,color=color[1],width=.4,label='ISFS',alpha=1)
        plt.title('Substitute Model ('+substitute[j]+')',fontdict={'size':40}) 
        plt.yticks(fontsize=40,rotation=0, horizontalalignment= 'right')
        plt.locator_params(nbins=5,axis='y')
        if j==0:
            plt.ylabel("Attack Success Rate (%)",fontsize=32)
            plt.legend(loc='upper center',ncol=5,bbox_to_anchor=(0.75,1),prop={'size':25},fancybox=True,shadow=False,edgecolor='black')
        plt.gca().set_xticklabels(x_label,fontdict={'size':35})
        plt.tight_layout()
    # plt.show()
    plt.savefig('/Users/zhanghangsheng/Desktop/TDSC/ETA/images/experiment/'+sheet_name+'-important.pdf',bbox_inches='tight', format='pdf', dpi=1000,transparent=False)

def plot_del_feature_line():
    # matplotlib.style.use('ggplot')
    df = pd.read_excel('/Users/zhanghangsheng/Desktop/实验数据/TIFS实验/3.1-删除非鲁棒特征.xlsx',sheet_name='ids17-botnet')
    print(df)
    x=df['dim']
    x_label = [1.5, 2.5, 3.5, 4.5, 5.5,6.5,7.5,8.5,9.5,10.5,11.5]
    color=['#f1595e','#ff1887','#8971e1','#007417','#56cefc']
    plt.figure(figsize=(10,5.8), dpi= 80)
    
    plt.plot(x,(1-df['LR'])*100,marker="o",markerfacecolor='white',linewidth=2,markersize=20,color=color[0],label='LR')
    plt.plot(x,(1-df['DT'])*100,marker="x",markerfacecolor='white',linewidth=2,markersize=20,color=color[1],label='DT')
    plt.plot(x,(1-df['SVM'])*100,marker='*',markerfacecolor='white',linewidth=2,markersize=20,color=color[2],label='SVM')
    plt.plot(x,(1-df['MLP'])*100,marker="^",markerfacecolor='white',linewidth=2,markersize=20,color=color[3],label='MLP')
    plt.plot(x,(1-df['Ens'])*100,marker="s",markerfacecolor='white',linewidth=2,markersize=20,color=color[4],label='Ens')

    plt.yticks(fontsize=32,rotation=0, horizontalalignment= 'right',)
    plt.gca().set_xticks(x)
    plt.gca().set_xticklabels(x, rotation=0, horizontalalignment= 'center',fontdict={'size':35})
    # plt.legend(loc='upper center',ncol=1,bbox_to_anchor=(0.8,0.995),prop={'size':20},fancybox=True,shadow=False)
    plt.xlabel("Percentage of Non-robust Features Removed (%)",fontsize=28)
    plt.ylabel("Attack Success Rate (%)",fontsize=30)
    plt.grid()
    plt.show()
    # plt.savefig('/Users/zhanghangsheng/Desktop/TIFS/ETA/images/experiment/del-ids17-botnet-line.pdf',bbox_inches='tight', format='pdf', dpi=1000,transparent=False)

def plot_use_feature_line():
    # matplotlib.style.use('ggplot')
    df = pd.read_excel('/Users/zhanghangsheng/Desktop/实验数据/TIFS实验/3.2-只用非鲁棒特征.xlsx',sheet_name='ids17-botnet')
    print(df)
    x=df['dim']
    x_label = [1.5, 2.5, 3.5, 4.5, 5.5,6.5,7.5,8.5,9.5,10.5,11.5]
    color=['#f1595e','#ff1887','#8971e1','#007417','#56cefc']
    plt.figure(figsize=(10,5.5), dpi= 80)
    
    plt.plot(x,df['LR'],marker="o",markerfacecolor='white',linewidth=2,markersize=20,color=color[0],label='LR')
    plt.plot(x,df['DT'],marker="x",markerfacecolor='white',linewidth=2,markersize=20,color=color[1],label='DT')
    plt.plot(x,df['SVM'],marker='*',markerfacecolor='white',linewidth=2,markersize=20,color=color[2],label='SVM')
    plt.plot(x,df['MLP'],marker="^",markerfacecolor='white',linewidth=2,markersize=20,color=color[3],label='MLP')
    plt.plot(x,df['Ens'],marker="s",markerfacecolor='white',linewidth=2,markersize=20,color=color[4],label='Ens')
    plt.ylim(0.5, 1.1)
    plt.yticks(fontsize=32,rotation=0, horizontalalignment= 'right',)
    plt.gca().set_xticks(x)
    plt.gca().set_xticklabels(x, rotation=0, horizontalalignment= 'center',fontdict={'size':35})
    plt.legend(loc='upper center',ncol=1,bbox_to_anchor=(0.8,0.995),prop={'size':35},fancybox=True,shadow=False)
    plt.xlabel("Percentage of Non-robust Features Used (%)",fontsize=30)
    plt.ylabel("Recall",fontsize=32)
    plt.grid()
    plt.show()
    # plt.savefig('/Users/zhanghangsheng/Desktop/TIFS/ETA/images/experiment/use-ids17-botnet-line.pdf',bbox_inches='tight', format='pdf', dpi=1000,transparent=False)

def plot_similar_line():
    # matplotlib.style.use('ggplot')
    df = pd.read_excel('/Users/zhanghangsheng/Desktop/实验数据/TIFS实验/3.3-不同模型之间的非鲁棒特征相似性.xlsx',sheet_name='ids17-ddos-1')
    print(df)
    # x=df['dim']
    x_label = [1.5, 2.5, 3.5, 4.5, 5.5,6.5,7.5,8.5,9.5,10.5,11.5]
    color=['#f1595e','#ff1887','#8971e1','#007417','#56cefc']
    plt.figure(figsize=(10,5.6), dpi= 80)
    
    plt.plot(df['LR-J'],df['LR-ASR']*100,marker="o",markerfacecolor='white',linewidth=2,markersize=10,color=color[0],label='LR')
    plt.plot(df['DT-J'],df['DT-ASR']*100,marker="x",markerfacecolor='white',linewidth=2,markersize=10,color=color[1],label='DT')
    plt.plot(df['SVM-J'],df['SVM-ASR']*100,marker='*',markerfacecolor='white',linewidth=2,markersize=10,color=color[2],label='SVM')
    plt.plot(df['MLP-J'],df['MLP-ASR']*100,marker="^",markerfacecolor='white',linewidth=2,markersize=10,color=color[3],label='MLP')
    plt.plot(df['Xgb-J'],df['Xgb-ASR']*100,marker="s",markerfacecolor='white',linewidth=2,markersize=10,color=color[4],label='Xgb')

    plt.yticks(fontsize=32,rotation=0, horizontalalignment= 'right')
    plt.xticks(fontsize=36)
    plt.legend(loc='upper center',ncol=5,bbox_to_anchor=(0.45,1.17),prop={'size':20},fancybox=True,shadow=False,edgecolor='black')
    plt.xlabel("Jaccard Similarity",fontsize=35)
    plt.ylabel("Attack Success Rate (%)",fontsize=32)
    plt.grid()
    # plt.show()
    plt.savefig('/Users/zhanghangsheng/Desktop/TIFS/ETA/images/leg/similar-leg.pdf',bbox_inches='tight', format='pdf', dpi=1000,transparent=False)
    # plt.savefig('/Users/zhanghangsheng/Desktop/TIFS/ETA/images/experiment/ids17-ddos-similar.pdf',bbox_inches='tight', format='pdf', dpi=1000,transparent=False)

def plot_sensitive():
    df = pd.read_excel('/Users/zhanghangsheng/Desktop/实验数据/TIFS实验/3.4-特征重要和敏感度.xlsx',sheet_name='ids17',index_col=0)
    df['senstive']=1-df['senstive']
    dic=df.to_dict('dict')['senstive']
    plt.figure(figsize=(7,5), dpi= 80)
    plt.barh(list(dic.keys()), list(dic.values()), color='#5e0e20')
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25,rotation=0, horizontalalignment= 'right')
    plt.xlabel("Sensitivity",fontsize=25)
    # plt.ylabel("Feature",fontsize=25)
    # plt.grid()
    # plt.savefig('/Users/zhanghangsheng/Desktop/TIFS/ETA/images/experiment/sensitive.pdf',bbox_inches='tight', format='pdf', dpi=1000,transparent=False)
    plt.show()

def plot_important():
    df = pd.read_excel('/Users/zhanghangsheng/Desktop/实验数据/TIFS实验/3.4-特征重要和敏感度.xlsx',sheet_name='ids17',index_col=0)
    df['important']=df['important']*10
    dic=df.to_dict('dict')['important']

    plt.figure(figsize=(7,5), dpi= 80)
    plt.barh(list(dic.keys()), list(dic.values()), color='#285085')
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25,rotation=0, horizontalalignment= 'right')
    plt.xlabel("Important",fontsize=25)
    plt.ylabel("Feature",fontsize=25)
    # plt.grid()
    # plt.show()
    plt.savefig('/Users/zhanghangsheng/Desktop/TIFS/ETA/images/experiment/important.pdf',bbox_inches='tight', format='pdf', dpi=1000,transparent=False)

def plot_feature_urg():
    x=[-0.5,0.5]
    color=['#1673ac','#b9232e']
    df = pd.read_excel('/Users/zhanghangsheng/Desktop/实验数据/TIFS实验/3.4-ids17-ddos-idle-max.xlsx',sheet_name='ids17-urg',index_col=0)
    print(df)
    plt.figure(figsize=(20,8), dpi= 80)
    plt.bar(x=[i + 0.5 for i in x], height=df['B']/40,color=color[0], width=0.15, label='Benign', alpha=1)
    plt.bar(x=[i + 0.5 for i in x], height=df['M']/40,color=color[1], width=0.15, label='Malicious', alpha=1)
    plt.xticks(fontsize=40)
    plt.yticks(fontsize=45,rotation=0, horizontalalignment= 'right')
    plt.gca().set_xticks(df['G'])
    plt.xlabel("URG Flag Count",fontsize=50)
    plt.ylabel("Distribution Frequency (%)",fontsize=45)
    plt.legend(loc='upper center',ncol=5,bbox_to_anchor=(0.55,1),prop={'size':45},edgecolor='black')
    plt.savefig('/Users/zhanghangsheng/Desktop/TIFS/ETA/images/experiment/urg.pdf',bbox_inches='tight', format='pdf', dpi=1000,transparent=False)
    # plt.show()

def plot_feature():
    x=[0,1,2,3,4,5,6,7,8,9,10,11,12,13]
    color=['#1673ac','#b9232e']
    df = pd.read_excel('/Users/zhanghangsheng/Desktop/实验数据/TIFS实验/3.4-ids17-ddos-idle-max.xlsx',sheet_name='ids17-iat',index_col=0)
    print(df)
    plt.figure(figsize=(25,6), dpi= 80)
    plt.bar(x=[i + 1 for i in x], height=df['B']/35,color=color[0], width=1, label='Benign', alpha=1)
    plt.bar(x=[i + 1 for i in x], height=df['M']/35,color=color[1], width=1, label='Malicious', alpha=1)
    plt.xticks(fontsize=60)
    plt.yticks(fontsize=60,rotation=0, horizontalalignment= 'right')
    plt.gca().set_xticks(df['G'])
    plt.xlabel("Fwd IAT Total (Group)",fontsize=55)
    # plt.xlabel("Min Packet Length (Group)",fontsize=50)
    # plt.xlabel("Idle Max (Group)",fontsize=50)
    # plt.ylabel("Distribution Frequency (%)",fontsize=45)
    # plt.legend(loc='upper center',ncol=5,bbox_to_anchor=(0.55,1),prop={'size':40},fancybox=True,shadow=True)
    # plt.savefig('/Users/zhanghangsheng/Desktop/TIFS/ETA/images/experiment/iat.pdf',bbox_inches='tight', format='pdf', dpi=1000,transparent=False)
    plt.savefig('/Users/zhanghangsheng/Desktop/TIFS/ETA/images/experiment/iat.pdf',bbox_inches='tight', format='pdf', dpi=1000,transparent=False)
    # plt.savefig('/Users/zhanghangsheng/Desktop/TIFS/ETA/images/experiment/iat.pdf',bbox_inches='tight', format='pdf', dpi=1000,transparent=False)
    # plt.show()



if __name__=='__main__':

    # plot_different_datasets()
    # plot_transfer()
    # plot_diff_feature()
    # plot_del_feature_line()
    plot_use_feature_line()
    # plot_similar_line()

    # plot_sensitive()
    # plot_important()
    # plot_feature()


