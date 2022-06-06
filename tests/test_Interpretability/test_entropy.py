import numpy as np
import sys
sys.path.append("/Users/zhanghangsheng/Documents/my_code/Cyber-Security-ML-Toolbox")
from csmt.get_model_data import get_datasets,parse_arguments
from csmt.figure.visualml.plot_importance import plot_feature_importance_all
np.seterr(divide='ignore', invalid='ignore')

def marginal_entropy(x,bins):
    px = np.histogram(x,bins)[0]/float(np.sum(np.histogram(x,bins)[0]))
    return np.nansum(px*np.log(1/px))

def joint_entropy(x,y,bins):
    pxy = np.histogram2d(x,y,bins)[0]/float(np.sum(np.histogram2d(x,y,bins)[0]))
    return np.nansum(pxy*np.log(1/pxy))

def mutual_information(x,y,bins):
    hx = marginal_entropy(x,bins)
    hy = marginal_entropy(y,bins)
    hxy = joint_entropy(x,y,bins)
    return hx+hy-hxy

def conditional_entropy(x,y,bins):
    hx = marginal_entropy(x,bins)
    hy = marginal_entropy(y,bins)
    hxy = joint_entropy(x,y,bins)
    return {'H(X|Y)': hxy-hy,
            'H(Y|X)': hxy-hx}

# x = np.random.normal(0, 1, 1000)
# y = np.random.normal(0, 1, 1000)
# bins = 20
# print('H(X) is {}'.format(marginal_entropy(x,bins)))
# print('H(X,Y) is {}'.format(joint_entropy(x,y,bins)))
# print('I(X,Y) is {}'.format(mutual_information(x,y,bins)))
# print('H(X|Y) is {}'.format(conditional_entropy(x,y,bins)['H(X|Y)']))
# print('H(Y|X) is {}'.format(conditional_entropy(x,y,bins)['H(Y|X)']))

if __name__=='__main__':
    arguments = sys.argv[1:]
    options = parse_arguments(arguments)
    datasets_name=options.datasets
    orig_models_name=options.algorithms
    evasion_algorithm=options.evasion_algorithm

    X_train,y_train,X_val,y_val,X_test,y_test,mask=get_datasets(options)

    bins=20
    # print(X_train[:,0])
    # print('H0(X,Y) is {}'.format(joint_entropy(X_train[:,0],y_train,bins)))
    # print('H1(X,Y) is {}'.format(joint_entropy(X_train[:,1],y_train,bins)))
    len_fea=X_train.shape[1]
    feature_importances=np.zeros(len_fea)
    for i in range(len_fea):
        feature_importances[i]=joint_entropy(X_train[:,i],y_train,bins)
    print(feature_importances)
    plot_feature_importance_all(feature_importances,max_num_features=10)


    