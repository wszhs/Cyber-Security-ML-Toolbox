import random
import numpy as np
import copy
from scapy.all import TCP, UDP, ICMP, IP, IPv6, Ether, ARP
from csmt.ps_attacks.traffic.unit import Unit
from csmt.ps_attacks.traffic.util import get_feature,get_new_feature,get_norm
from csmt.ps_attacks.traffic.rebulid_pcap import rebuild
from csmt.get_model_data import models_predict
from csmt.zoopt.bayes_opt.bayesian_optimization import BayesianOptimization

def eval(inter_arr,groupList,grp_size,norm,models):
    new_groupList=rebuild(grp_size, inter_arr, groupList)
    new_feature=get_new_feature(new_groupList, inter_arr)
    new_feature=norm.transform(new_feature)
    y_fea=np.ones(grp_size)
    y_fea,y_pred_new=models_predict(models,new_feature,y_fea)
    y_pred_new=np.argmax(y_pred_new[0], axis=1)
    from sklearn import metrics
    recall_new= metrics.recall_score(y_fea, y_pred_new)
    return recall_new

def time_pertur(X,groupList,grp_size,last_end_time,per_vec):
    per_times=0  #扰动的倍数 整数
    ics_time = 0  # accumulated increased ITA
    for i in range(grp_size):
        if i == 0:
            itv = groupList[i].time - last_end_time
        else:
            itv = groupList[i].time - groupList[i - 1].time
        if i in per_vec[:,0]:
            per_times=per_vec[per_vec[:,0]==i,1][0]
        ics_time+=per_times*itv
        X.mal[i][0] = groupList[i].time + ics_time
    return X

def adaptive_time_pertur(inter_arr,groupList,grp_size,norm,models,last_end_time,max_cft_pkt):

    def get_score(p):
        # 重新初始化inter
        inter_arr= Unit(grp_size, max_cft_pkt)
        per_vec=-1*np.ones((budget,2))
        for i in range(budget):
            per_vec[i,:]=[int(p[2*i]),int(p[2*i+1])]
        per_times=0  #扰动的倍数 整数
        ics_time = 0  # accumulated increased ITA
        for i in range(grp_size):
            if i == 0:
                itv = groupList[i].time - last_end_time
            else:
                itv = groupList[i].time - groupList[i - 1].time
            if i in per_vec[:,0]:
                per_times=per_vec[per_vec[:,0]==i,1][0]
            ics_time+=per_times*itv
            inter_arr.mal[i][0] = groupList[i].time + ics_time
        score=eval(inter_arr,groupList,grp_size,norm,models)
        return -score
    # 扰动的budget
    budget=2
    max_time_extend=20
    min_time_extend=3

    arr=np.zeros((budget*2,2))
    keys=[]
    for i in range(budget):
        arr[2*i,:]=[0,grp_size]
        arr[2*i+1,:]=[min_time_extend,max_time_extend]
        keys.append('x'+str(2*i))
        keys.append('x'+str(2*i+1))
    optimizer = BayesianOptimization(get_score,{'x':arr},random_state=7)
    optimizer.maximize(init_points=5, n_iter=10)
    max_x=np.array([optimizer.max['params'][key] for key in keys])

    per_vec=-1*np.ones((budget,2))
    for i in range(budget):
        per_vec[i,:]=[int(max_x[2*i]),int(max_x[2*i+1])]
    # 重新初始化inter
    inter_arr= Unit(grp_size, max_cft_pkt)
    inter_arr=time_pertur(inter_arr,groupList,grp_size,last_end_time,per_vec)
    return inter_arr

def size_pertur(X,grp_size,last_end_time,groupList,max_cft_pkt,per_vec):
    
    max_time_extend=10
    max_mal_itv = (groupList[-1].time - last_end_time) * (max_time_extend + 1)
    # building slot map
    slot_num = grp_size * max_cft_pkt
    slot_itv = max_mal_itv / slot_num

    nxt_mal_no = 0
    proto_max_lmt = []  # maximum protocol layer number
    for i in range(grp_size):
        if groupList[i].haslayer(TCP) or groupList[i].haslayer(UDP) or groupList[i].haslayer(ICMP):
            proto_max_lmt.append(3.)
        elif groupList[i].haslayer(IP) or groupList[i].haslayer(IPv6) or groupList[i].haslayer(ARP):
            proto_max_lmt.append(2.)
        elif groupList[i].haslayer(Ether):
            proto_max_lmt.append(1.)
        else:
            proto_max_lmt.append(0.)

    for i in range(slot_num):
        slot_time = i * slot_itv + last_end_time
        if slot_time >= X.mal[nxt_mal_no][0]:
            nxt_mal_no += 1
            if nxt_mal_no == grp_size:
                break

        if (not i in per_vec[:,0]) or X.mal[nxt_mal_no][1] == max_cft_pkt:
            continue

        cft_no = int(round(X.mal[nxt_mal_no][1]))
        if proto_max_lmt[nxt_mal_no] == 3.:
            X.craft[nxt_mal_no][cft_no][1] = 3
            mtu = 1460
        elif proto_max_lmt[nxt_mal_no] == 2.:
            X.craft[nxt_mal_no][cft_no][1] = 2
            mtu = 1480
        elif proto_max_lmt[nxt_mal_no] == 1.:
            X.craft[nxt_mal_no][cft_no][1] = 1
            mtu = 1500
        else:
            continue
        X.craft[nxt_mal_no][cft_no][0] = X.mal[nxt_mal_no][0] - slot_time
        X.craft[nxt_mal_no][cft_no][2] = np.min([per_vec[per_vec[:,0]==i,1][0],mtu])
        X.mal[nxt_mal_no][1] += 1.
    return X

def adaptive_size_pertur(inter_arr,grp_size,last_end_time,groupList,max_cft_pkt,norm,models):
    # 产生对抗包的数组
    def get_score(p):
        # 重新初始化inter
        inter_arr= Unit(grp_size, max_cft_pkt) 
        per_vec=-1*np.ones((budget,2))
        for i in range(budget):
            per_vec[i,:]=[int(p[2*i]),int(p[2*i+1])]
        # 修改size
        max_time_extend=10
        max_mal_itv = (groupList[-1].time - last_end_time) * (max_time_extend + 1)
        # building slot map
        slot_num = grp_size * max_cft_pkt
        slot_itv = max_mal_itv / slot_num

        nxt_mal_no = 0
        proto_max_lmt = []  # maximum protocol layer number
        for i in range(grp_size):
            if groupList[i].haslayer(TCP) or groupList[i].haslayer(UDP) or groupList[i].haslayer(ICMP):
                proto_max_lmt.append(3.)
            elif groupList[i].haslayer(IP) or groupList[i].haslayer(IPv6) or groupList[i].haslayer(ARP):
                proto_max_lmt.append(2.)
            elif groupList[i].haslayer(Ether):
                proto_max_lmt.append(1.)
            else:
                proto_max_lmt.append(0.)

        for i in range(slot_num):
            slot_time = i * slot_itv + last_end_time
            if slot_time >= inter_arr.mal[nxt_mal_no][0]:
                nxt_mal_no += 1
                if nxt_mal_no == grp_size:
                    break

            if (not i in per_vec[:,0]) or inter_arr.mal[nxt_mal_no][1] == max_cft_pkt:
                continue
            cft_no = int(round(inter_arr.mal[nxt_mal_no][1]))
            if proto_max_lmt[nxt_mal_no] == 3.:
                inter_arr.craft[nxt_mal_no][cft_no][1] =3
                mtu = 1460
            elif proto_max_lmt[nxt_mal_no] == 2.:
                inter_arr.craft[nxt_mal_no][cft_no][1] = 2
                mtu = 1480
            elif proto_max_lmt[nxt_mal_no] == 1.:
                inter_arr.craft[nxt_mal_no][cft_no][1] = 1
                mtu = 1500
            else:
                continue
            inter_arr.craft[nxt_mal_no][cft_no][0] = inter_arr.mal[nxt_mal_no][0] - slot_time
            inter_arr.craft[nxt_mal_no][cft_no][2] = np.min([per_vec[per_vec[:,0]==i,1][0],mtu])
            inter_arr.mal[nxt_mal_no][1] += 1.
        score=eval(inter_arr,groupList,grp_size,norm,models)
        return -score

    budget=10
    budget_size=1000
    arr=np.zeros((budget*2,2))
    keys=[]
    for i in range(budget):
        arr[2*i,:]=[0,grp_size*max_cft_pkt]
        arr[2*i+1,:]=[budget_size,budget_size]
        keys.append('x'+str(2*i))
        keys.append('x'+str(2*i+1))
    optimizer = BayesianOptimization(get_score,{'x':arr},random_state=7)
    optimizer.maximize(init_points=5, n_iter=10)
    max_x=np.array([optimizer.max['params'][key] for key in keys])

    per_vec=-1*np.ones((budget,2))
    for i in range(budget):
        per_vec[i,:]=[int(max_x[2*i]),int(max_x[2*i+1])]
    # 重新初始化inter
    inter_arr= Unit(grp_size, max_cft_pkt) 
    inter_arr=size_pertur(inter_arr,grp_size,last_end_time,groupList,max_cft_pkt,per_vec)
    return inter_arr

# 生成对抗流
def bayes_attack(grp_size,last_end_time,groupList,norm,models):
    max_cft_pkt=2
    inter_arr= Unit(grp_size, max_cft_pkt)  # position vector
    inter_arr=adaptive_time_pertur(inter_arr,groupList,grp_size,norm,models,last_end_time,max_cft_pkt)
    # inter_arr=adaptive_size_pertur(inter_arr,grp_size,last_end_time,groupList,max_cft_pkt,norm,models)
    return inter_arr