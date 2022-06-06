__author__ = 'P-F.Marteau, June 2020'

import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np
import time
import pickle
import pathlib
    
from sklearn.metrics import roc_curve, auc
from sklearn.ensemble import IsolationForest

from DiFF_RF import DiFF_TreeEnsemble

plt.gcf().subplots_adjust(bottom=0.15)
matplotlib.rcParams.update({'font.size': 22})

    
def gen_tore_vecs(dims, number, rmin, rmax):
    vecs = np.random.uniform(low=-1, size=(number,dims))
    radius = rmin + np.random.sample(number) * (rmax-rmin)
    mags = np.sqrt((vecs*vecs).sum(axis=-1))
    # How to distribute the magnitude to the vectors
    for i in range(number):
        vecs[i,:] = vecs[i, :] / mags[i] *radius[i]
    return vecs[:,0], vecs[:,1]


def createDonutData(contamin=0):
    print('build donnuts data')
    Nobjs = 1000
    xn, yn = gen_tore_vecs(2, Nobjs, 1.5, 4)
    Xn = np.array([xn, yn]).T

    Nobjsb = 1000
    mean = [0, 0]
    cov = [[.5, 0], [0, .5]]  # diagonal covariance
    xb, yb = np.random.multivariate_normal(mean, cov, Nobjsb).T
    Xb = np.array([xb, yb]).T

    Nobjst = 1000
    xnt, ynt = gen_tore_vecs(2, Nobjst, 1.5, 4)
    Xnt = np.array([xnt, ynt]).T
    

    # create cluster of anomalies
    mean = [3., 3.]
    cov = [[.25, 0], [0, .25]]  # diagonal covariance
    Nobjsa = 1000
    xa, ya = np.random.multivariate_normal(mean, cov, Nobjsa).T
    Xa = np.array([xa, ya]).T
    
    Xab=np.concatenate([Xa,Xb])

    pathlib.Path('tests/example/PKL').mkdir(parents=True, exist_ok=True) 
    f = open('tests/example/PKL/donnutsDataProblem.pkl', 'wb')
    pickle.dump([Xn, Xnt, Xa, Xb, Xab], f)
    f.close()


def computeDiff_RF(ntrees=1024, sample_size_ratio=.33, alpha0=.1):
    # load data
    f = open('tests/example/PKL/donnutsDataProblem.pkl', 'rb')
    [Xn, Xnt, Xa, Xb, Xab] = pickle.load(f)

    f.close()
    
    if sample_size_ratio >1:
        sample_size=sample_size_ratio
    else:
        sample_size=int(sample_size_ratio*len(Xn))

    xn=Xn[:,0]
    yn=Xn[:,1]
    xa=Xa[:,0]
    ya=Xa[:,1]

    xb=Xb[:,0]
    yb=Xb[:,1]

    # pathlib.Path('tests/example/FIG').mkdir(parents=True, exist_ok=True) 
    # # plotting the donnuts data
    # plt.figure(1)
    # plt.plot(xn, yn, 'bo', markersize=10)
    # plt.savefig('tests/example/FIG/clustersDonnuts0.pdf')

    # nn=len(Xa)
    # plt.figure(2)
    # plt.plot(xn, yn, 'bo', xa[0:nn], ya[0:nn], 'rs')
    # plt.savefig('tests/example/FIG/clustersDonnuts1.pdf')

    # plt.figure(3)
    # plt.plot(xn, yn, 'bo', xa[0:nn], ya[0:nn], 'rs', xb[0:nn], yb[0:nn], 'gd')
    # plt.xticks(size=14)
    # plt.yticks(size=14)
    # plt.savefig('tests/example/FIG/clustersDonnuts2.pdf')

    # Creating Forest on normal data + anomalies labels
    print('building the Diff_RF ...')

    diff_rf = DiFF_TreeEnsemble(sample_size=sample_size, n_trees=ntrees)    # load data
    fit_start = time.time()
    diff_rf.fit(Xn, n_jobs=8)
    fit_stop = time.time()
    fit_time = fit_stop - fit_start
    print(f"fit time {fit_time:3.2f}s")
    n_nodes = sum([t.n_nodes for t in diff_rf.trees])
    print(f"{n_nodes} total nodes in {ntrees} trees")
    
    XT=np.concatenate([Xnt,Xab])
    
    sc_di,sc_ff,sc_diff_rf = diff_rf.anomaly_score(XT,alpha=alpha0)
    sc_diff_rf=np.array(sc_diff_rf)
    sc_ff=np.array(sc_ff)
    sc_di=np.array(sc_di)
    sc_ff=(sc_ff-sc_ff.min())/(sc_ff.max()-sc_ff.min())
    sc_di=(sc_di-sc_di.min())/(sc_di.max()-sc_di.min())
    sc_diff_rf=(sc_diff_rf-sc_diff_rf.min())/(sc_diff_rf.max()-sc_diff_rf.min())

    plt.figure(1000)
    xn=XT[:,0]
    yn=XT[:,1]
    plt.scatter(xn, yn, marker='o', c=sc_ff, cmap='viridis')
    plt.colorbar()
    plt.xticks(size=14)
    plt.yticks(size=14)
    plt.title('DiFF_RF (visiting frequency score) Heat Map')
    plt.savefig('tests/example/FIG/HeatMap_DiFF_RF_freqScore.pdf')
    
    plt.figure(1001)
    xn=XT[:,0]
    yn=XT[:,1]
    plt.scatter(xn, yn, marker='o', c=sc_diff_rf, cmap='viridis')
    plt.colorbar()
    plt.xticks(size=14)
    plt.yticks(size=14)
    plt.title('DiFF_RF (collective anomaly score) Heat Map')
    plt.savefig('tests/example/FIG/HeatMap_DiFF_RF_collectiveScore.pdf')
    
    plt.figure(1002)
    xn=XT[:,0]
    yn=XT[:,1]
    plt.scatter(xn, yn, marker='o', c=(sc_di), cmap='viridis')
    plt.colorbar()
    plt.xticks(size=14)
    plt.yticks(size=14)
    plt.title('DiFF_RF (point-wise anomaly score) Heat Map')
    plt.savefig('tests/example/FIG/HeatMap_DiFF_RF_pointWiseScore.pdf')

    cif = IsolationForest(n_estimators=ntrees, max_samples=sample_size, bootstrap=False, n_jobs=12)
    cif.fit(Xn)
    sc_if = -cif.decision_function(XT)
    sc_if=(sc_if-sc_if.min())/(sc_if.max()-sc_if.min())
    plt.figure(1003)
    xn=XT[:,0]
    yn=XT[:,1]
    plt.scatter(xn, yn, marker='o', c=sc_if, cmap='viridis')
    plt.colorbar()
    plt.xticks(size=14)
    plt.yticks(size=14)
    plt.title('Isolation Forest Heat Map')
    plt.savefig('tests/example/FIG/HeatMap_IF.pdf')
    # plt.show()
    
    y_true = np.array([-1] * len(Xnt) + [1] * len(Xab))

    fpr_IF, tpr_IF, thresholds = roc_curve(y_true, sc_if)
    aucIF=auc(fpr_IF, tpr_IF)
    fpr_D, tpr_D, thresholds = roc_curve(y_true, sc_di)
    aucD=auc(fpr_D, tpr_D)
    fpr_F, tpr_F, thresholds = roc_curve(y_true, sc_ff)
    aucF=auc(fpr_F, tpr_F)
    fpr_DF, tpr_DF, thresholds = roc_curve(y_true, sc_diff_rf)
    aucDF=auc(fpr_DF, tpr_DF)
    print("Isolation Forest AUC=", aucIF)
    print("DiFF_RF (point-wise anomaly score) AUC=", aucD)
    print("DiFF_RF (frequency of visit scoring only) AUC=", aucF)
    print("DiFF_RF (collective anomaly score) AUC=", aucDF)


# create donnuts data
createDonutData(contamin=0)

# build and test IF and DiFF-RF
computeDiff_RF(ntrees=256, sample_size_ratio=.25, alpha0=1)
