import sys

from pandas.core import base
sys.path.append("/Users/zhanghangsheng/Documents/my_code/Cyber-Security-ML-Toolbox")
import matplotlib
import csmt.Interpretability.sage as sage
import os

import matplotlib.pyplot as plt
import csmt.figure.visualml.visualml as vml
import pandas as pd
from csmt.get_model_data import get_datasets,parse_arguments,models_train,print_results,models_predict
import numpy as np
import torch
import seaborn as sns
import csmt.Interpretability.shap as shap

def plot_feature_importance_all(importance, ax=None, height=0.5,
                    xlim=None, ylim=None, title='Feature importance',
                    xlabel='F score', ylabel='Features', max_num_features=None,
                    grid=True, show_values=True, **kwargs):

    if not importance:
        raise ValueError('Booster.get_score() results in empty')

    tuples = [(k, importance[k]) for k in importance]
    if max_num_features is not None:
        # pylint: disable=invalid-unary-operand-type
        tuples = sorted(tuples, key=lambda x: x[1])[-max_num_features:]
    else:
        tuples = sorted(tuples, key=lambda x: x[1])
    labels, values = zip(*tuples)

    if ax is None:
        _, ax = plt.subplots(1, 1)

    ylocs = np.arange(len(values))
    ax.barh(ylocs, values, align='center', height=height, **kwargs)

    if show_values is True:
        for x, y in zip(values, ylocs):
            ax.text(x, y, x, va='center')

    ax.set_yticks(ylocs)
    ax.set_yticklabels(labels)

    if xlim is not None:
        if not isinstance(xlim, tuple) or len(xlim) != 2:
            raise ValueError('xlim must be a tuple of 2 elements')
    else:
        xlim = (0, max(values) * 1.1)
    ax.set_xlim(xlim)

    if ylim is not None:
        if not isinstance(ylim, tuple) or len(ylim) != 2:
            raise ValueError('ylim must be a tuple of 2 elements')
    else:
        ylim = (-1, len(values))
    ax.set_ylim(ylim)

    if title is not None:
        ax.set_title(title)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    ax.grid(grid)
    # plt.show()
    plt.savefig('experiments/important/'+datasets_name+'/'+options.attack_models+'/'+title+'.pdf', format='pdf',bbox_inches='tight',dpi=1000,transparent=True)
    return ax


if __name__=='__main__':
    arguments = sys.argv[1:]
    options = parse_arguments(arguments)
    datasets_name=options.datasets
    orig_models_name=options.algorithms
    evasion_algorithm=options.evasion_algorithm

    X_train,y_train,X_val,y_val,X_test,y_test,mask=get_datasets(options)

    trained_models=models_train(datasets_name,orig_models_name,False,X_train,y_train,X_val,y_val)

    y_test,y_pred=models_predict(trained_models,X_test,y_test)
    print_results(datasets_name,orig_models_name,y_test,y_pred,'original_accuracy')

    imputer = sage.MarginalImputer(trained_models[0],X_test[:100])
    estimator = sage.PermutationEstimator(imputer, 'cross entropy')
    # estimator = sage.IteratedEstimator(imputer, 'cross entropy')
    sage_values = estimator(X_test, y_test)

    sage_values.plot(max_features=20)
    print(sage_values.values)
    plt.show()

    # path='experiments/important/'+datasets_name+'/'+options.attack_models
    # isExists=os.path.exists(path)
    # if not isExists:
    #     os.makedirs(path) 
    # plt.savefig(path+'/ddos_important.pdf', format='pdf',bbox_inches='tight',dpi=1000,transparent=True)

    # argsort = np.argsort(sage_values.values)[::-1]
    # group_cor=list(combinations(argsort[:5], 2))
    # group_single=list(combinations(argsort[:5], 1))
    # groups=group_cor+group_single
    # group_names = [str(i).replace('(','').replace(')','').replace(',','') for i in groups]

    # imputer = sage.GroupedMarginalImputer(attack_model,X_test[:100],groups)
    # # estimator = sage.PermutationEstimator(imputer, 'cross entropy')
    # estimator = sage.IteratedEstimator(imputer, 'cross entropy')
    # # estimator = sage.KernelEstimator(imputer, 'cross entropy')
    # sage_values = estimator(X_test, y_test)
    # # print(sage_values.values)

    # dic_sage = dict(zip(group_names,sage_values.values))
    # dic_cor=dict(zip(group_names[:len(group_cor)],sage_values.values[:len(group_cor)]))
    # for key in dic_cor.keys():
    #     dic_cor[key]=dic_sage[key]-(dic_sage[key.split(' ')[0]]+dic_sage[key.split(' ')[1]])
    # print(dic_sage)
    # print(dic_cor)

    # plot_feature_importance_all(dic_sage,title='Feature importance')
    # plot_feature_importance_all(dic_cor,title='Cor importance',show_values=False)
            
    # sage_values.plot(group_names)
    # plt.show()


