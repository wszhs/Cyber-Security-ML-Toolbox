import matplotlib
matplotlib.use('TKAgg')
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def plot_dot(X,a_score):
    X_x=X[:,0]
    X_y=X[:,1]
    plt.scatter(X_x, X_y, marker='o', c=a_score, cmap='viridis')
    plt.colorbar()
    plt.xticks(size=14)
    plt.yticks(size=14)
    plt.title('dot important')
    plt.show()

def plot_feature_importance_all(feature_importances, ax=None, height=0.2,
                    xlim=None, ylim=None, title='Feature importance',
                    xlabel='F score', ylabel='Features',
                    importance_type='gain', max_num_features=None,
                    grid=True, show_values=True, **kwargs):

    # """Get feature importance of each feature.
    #     Importance type can be defined as:

    #     * 'weight': the number of times a feature is used to split the data across all trees.
    #     * 'gain': the average gain across all splits the feature is used in.
    #     * 'cover': the average coverage across all splits the feature is used in.
    #     * 'total_gain': the total gain across all splits the feature is used in.
    #     * 'total_cover': the total coverage across all splits the feature is used in.

    feature_names=[]
    for i in range(len(feature_importances)):
        feature_names.append('f'+str(i))

    feature_ = list(zip(feature_names,map(lambda x: round(x, 4), feature_importances)))
    importance=dict(feature_)

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
            ax.text(x+0.01, y, x, va='center')

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
    plt.show()
    return ax


def plot_xg_importance(booster, ax=None, height=0.2,
                    xlim=None, ylim=None, title='Feature importance',
                    xlabel='F score', ylabel='Features',
                    importance_type='weight', max_num_features=None,
                    grid=True, show_values=True, **kwargs):

    # """Get feature importance of each feature.
    #     Importance type can be defined as:

    #     * 'weight': the number of times a feature is used to split the data across all trees.
    #     * 'gain': the average gain across all splits the feature is used in.
    #     * 'cover': the average coverage across all splits the feature is used in.
    #     * 'total_gain': the total gain across all splits the feature is used in.
    #     * 'total_cover': the total coverage across all splits the feature is used in.

    importance = booster.get_booster().get_score(importance_type=importance_type)

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
            ax.text(x + 1, y, x, va='center')

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
    plt.show()
    return ax

# def plot_feature_importance(feature_importances):
#     feature_names=[]
#     for i in range(len(feature_importances)):
#         feature_names.append('f'+str(i))
#     feature = list(zip(map(lambda x: round(x, 4), feature_importances), feature_names))

#     imp_names = []
#     imp_values = []
#     for i in feature:
#         if i[0] != 0.0:
#             imp_names.append(i[1])
#             imp_values.append(i[0])
#         else:
#             pass 

#     length = np.arange(len(imp_names))

#     plt.barh(length, imp_values, align='center', alpha=0.5)
#     plt.yticks(length, imp_names)
#     plt.ylabel('Feature name')
#     plt.xlabel('Feature importance')
#     plt.show()