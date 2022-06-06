
import sys
sys.path.append("/Users/zhanghangsheng/Documents/my_code/Cyber-Security-ML-Toolbox")
import numpy as np
from scapy.all import *
from csmt.get_model_data import get_datasets,parse_arguments,models_train,print_results,models_predict,models_load,get_raw_datasets
from scapy.all import rdpcap,wrpcap,hexdump
from csmt.ps_attacks.traffic.rebulid_pcap import rebuild
from csmt.ps_attacks.traffic.util import get_feature,get_new_feature,get_norm
from csmt.ps_attacks.packet_attack.RandomAttack import random_attack
from csmt.ps_attacks.packet_attack.BayesAttack import bayes_attack

def EvasionAttack(last_end_time,groupList,grp_size,norm,models):

    # inter_arr=random_attack(grp_size,last_end_time,groupList)
    inter_arr=bayes_attack(grp_size,last_end_time,groupList,norm,models)
    new_groupList=rebuild(grp_size, inter_arr, groupList)
    cur_end_time = inter_arr.mal[-1][0]
    ics_time = cur_end_time - float(groupList[-1].time)

    return new_groupList,inter_arr,cur_end_time,ics_time

def load_orig_pcap():
    pcap_file='tests/example/malware.pcap'
    pktList = rdpcap(pcap_file)
    last_end_time = float(pktList[0].time)
    return last_end_time,pktList


if __name__=='__main__':
    arguments = sys.argv[1:]
    options = parse_arguments(arguments)
    datasets_name=options.datasets
    orig_models_name=options.algorithms
    X,y,mask=get_raw_datasets(options)

    # 初始化归一化器
    norm=get_norm(X)
    # 加载模型
    models=models_load(datasets_name,orig_models_name)
    # X=norm.transform(X)
    # y_test,y_pred=models_predict(models,X,y)
    # table=print_results(datasets_name,orig_models_name,y_test,y_pred,'original_accuracy')

    # # 加载恶意pcap包
    last_end_time,pktList=load_orig_pcap()
    pcap_size=1000
    old_fea=get_feature(pktList)
    new_fea=copy.deepcopy(old_fea)
    y_fea=np.ones(old_fea.shape[0])

    grp_size=1000
    cycle_time=int(pcap_size/grp_size)
    acc_ics_time=0
    st = 0
    ed = st + grp_size
    for i in range(cycle_time):
        groupList = pktList[st:ed]
        for pkt in groupList:
            pkt.time = float(pkt.time) + acc_ics_time
        # # 开始规避攻击
        new_groupList,inter_arr,cur_end_time,ics_time=EvasionAttack(last_end_time,groupList,grp_size,norm,models)
        # 特征提取
        new_grp_fea=get_new_feature(new_groupList,inter_arr)
        new_fea[st:ed]=new_grp_fea
        
        acc_ics_time += ics_time
        last_end_time = cur_end_time
        st = ed
        ed += grp_size

    # # 归一化特征向量
    old_fea=norm.transform(old_fea)
    new_fea=norm.transform(new_fea)

    y_fea,y_pred_old=models_predict(models,old_fea,y_fea)
    y_pred_old=np.argmax(y_pred_old[0], axis=1)

    y_fea,y_pred_new=models_predict(models,new_fea,y_fea)
    y_pred_new=np.argmax(y_pred_new[0], axis=1)

    # print(y_pred_old[0:1000])
    # print(y_pred_new[0:1000])

    from sklearn import metrics
    recall_old = metrics.recall_score(y_fea, y_pred_old)
    recall_new= metrics.recall_score(y_fea, y_pred_new)

    print(recall_old)
    print(recall_new)

