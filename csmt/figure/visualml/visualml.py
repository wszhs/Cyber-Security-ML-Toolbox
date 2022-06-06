# -*- coding: utf-8 -*-
from __future__ import division
import matplotlib
matplotlib.use('TKAgg')
import seaborn as sns
sns.set_context("notebook", font_scale=1.1)
sns.set_style("ticks")
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


__all__ = ['decision_boundary_grid', 'plot_decision_boundary']

def map_decision_boundary(clf,X, target, ax=None, plt_title=''):
    feature_names=[]
    for i in range(X.shape[1]):
        feature_names.append('comp'+str(i+1))
    X = pd.DataFrame(X, columns=feature_names)
    feature_A = X[feature_names[0]].values
    feature_B = X[feature_names[1]].values
    # Get limits of the mesh grid
    x_min, x_max = min(feature_A), max(feature_A)
    y_min, y_max = min(feature_B), max(feature_B)
    
    # Get step of the mesh grid
    h_x = (x_max - x_min)/10
    h_y = (y_max - y_min)/10
    
    # Create the meshgrid
    xx, yy = np.meshgrid(np.arange(x_min, x_max+h_x, h_x),
                         np.arange(y_min, y_max+h_x, h_y))
    
    # Get prediction values (either probabilities or from decision function)
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])[:, 0]
    
    # Convert Z to 2D
    Z = Z.reshape(xx.shape)
    
    # Plot contour function    
    cm_bright = ListedColormap(['#0000FF', '#FF0000'])
    if ax is None:
        ax=plt.gca()
    ax.contourf(xx, yy, Z, cmap=plt.cm.RdBu, alpha=.8)
    ax.scatter(feature_A, feature_B, c=target, cmap=cm_bright, edgecolors='k')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    plt.xlabel(feature_names[0])
    plt.ylabel(feature_names[1])
    plt.title(plt_title)
    plt.show()

def decision_boundary_grid(clf, orig_model_name,X, y, feat_list=None, cmap_bkg='RdBu', \
    color_labels=['#0000FF', '#FF0000'], **fig_kw):
    """
    Creates a pairwise grid of decision boundaries from all the combination of
    pairs of features. 

    Parameters
    ----------
    clf : estimator object
        Classifier estimator implemented using the scikit-learn interface 

    X : pandas dataframes
        Set of features 

    y : labels

    Examples
    --------
    >>> import visualml as vml
    >>> import pandas as pd
    >>> from sklearn.datasets import make_classification
    >>> from sklearn.ensemble import RandomForestClassifier as RF
    >>>
    >>> # Create a toy classification dataset
    >>> feature_names = ['A','B','C','D']
    >>> X, y = make_classification(n_features=4, random_state=42)
    >>>
    >>> # The visualization is only supported if X is a pandas df
    >>> X = pd.DataFrame(X, columns=feature_names)
    >>>
    >>> # Train a classifier
    >>> clf = RF(random_state=42).fit(X,y)
    >>>
    >>> # Plot decision boundary grid
    >>> vml.decision_boundary_grid(clf, X, y)

    """
    ### Plot main diagonal
    # Get the number of columns (features)
    feature_names=[]
    for i in range(X.shape[1]):
        feature_names.append('comp'+str(i))
    X = pd.DataFrame(X, columns=feature_names)
    if feat_list is None:
        feat_list = X.columns

    n_cols = len(feat_list)
    fig, ax = plt.subplots(n_cols, n_cols, \
        gridspec_kw = {'wspace':0.07, 'hspace':0.07}, **fig_kw)
    # fig.suptitle(orig_model_name)
    ### Plot off diagonals
    for i_x, col_x in enumerate(feat_list):
        for i_y, col_y in enumerate(feat_list):
            if isinstance(ax,np.ndarray):
                ax_i = ax[i_y][i_x] # Row first, which is the Y axis
            else:
                ax_i = ax # If feat_list has only one input

            # Call function to plot pairs of attributes
            plot_decision_boundary(clf, X, y, [col_x, col_y], \
                ax=ax_i, cmap_bkg=cmap_bkg)

            ax_i.get_xaxis().set_visible(False)
            ax_i.get_yaxis().set_visible(False)

            if i_y==n_cols-1:
                # ax_i.get_xaxis().set_visible(True)
                ax_i.tick_params(axis='x', labelsize=7)
                ax_i.set_xlabel(col_x)
            if i_x==0:
                # ax_i.get_yaxis().set_visible(True)
                # ax_i.tick_params(axis='y', labelsize=7)
                ax_i.set_ylabel(col_y)

    plt.show()
    # plt.savefig('/Users/zhanghangsheng/Desktop/TIFS/ETA/images/experiment/dim_botnet_lr.pdf',bbox_inches='tight', format='pdf', dpi=1000,transparent=True)
    #return fig



def plot_decision_boundary(clf, X, y, cols, ax=None, cmap_bkg='RdBu', \
    color_labels=['#0000FF', '#FF0000']):
    """
    Plots the decision boundary's 2D or 1D projection of a trained model. When 
    the input is a pair of features, the decision boundary is visualized in the 
    background of a scatter plot. When the input is one single feature, the 
    decision boundary is visualized in the background of a histogram plot. 

    Parameters
    ----------
    clf : estimator object
        Classifier estimator implemented using the scikit-learn interface. 

    X : pandas dataframes
        Set of features 

    y : labels

    cols : list, str
        Name of the single or pair of features to be visualized

    ax : matplotlib axis
        axis object


    Examples
    --------
    >>> from sklearn.datasets import make_classification
    >>> X, y = make_classification(n_features=4, random_state=42)
    >>> from sklearn.svm import SVC
    >>> clf = SVC(random_state=42).fit(X,y)
    >>> import pandas as pd
    >>> X = pd.DataFrame(X, columns=['A','B','C','D'])
    >>> plot_decision_boundary(clf, X, y, 'A')

    """
    # Get color maps for the labels and background
    cm = plt.get_cmap(cmap_bkg)
    cmap_labels = ListedColormap(color_labels)

    # Check if the set of features X is a pandas dataframe
    if not isinstance(X, pd.DataFrame):
        _raise_error("The input set of features X should be a Pandas DataFrame")
        return

    # Check and/or solve possible bugs in the dimension
    cols=_check_cols_bugs(cols)

    # Main task: plot hist if 1D of plot scatter if 2D
    if len(cols)==2: # 2D plot (scatter)
        # Get mesh grid values
        xx, yy, Z = _get_mesh_coordinates(clf, X, y, cols)
        # If axis ax is defined in the function (from the subplot), use it
        if ax==None:
            plt.contourf(xx,yy,Z, cmap=cm)
            plt.scatter(X[cols[0]], X[cols[1]],s=10,c=y, cmap=cmap_labels, alpha=.3)
            ax=plt.gca()
            ax.set_xlim(np.min(xx), np.max(xx))
            ax.set_ylim(np.min(yy), np.max(yy))
            plt.show()
        else:
            ax.contourf(xx,yy,Z, cmap=cm)
            ax.scatter(X[cols[0]], X[cols[1]],s=10,c=y, cmap=cmap_labels, alpha=.3)
            ax.set_xlim(np.min(xx), np.max(xx))
            ax.set_ylim(np.min(yy), np.max(yy))

    else: # 1D plot (histograms)
        # List of each group class values in each column
        X_hist = []
        for group in set(y):
            X_hist.append(X[cols][y==group])

        if ax==None: # Check if it is part of subplot or not
            hist_values = plt.hist(X_hist, stacked=True, alpha=.7, color=color_labels)
            xx, yy, Z = _get_mesh_coordinates(clf, X, y, cols, hist_values)
            plt.contourf(xx,yy,Z, cmap=cm)
            plt.show()
        else: # in case it is part of subplot
            hist_values = ax.hist(X_hist, stacked=True, alpha=.7, color=color_labels)
            xx, yy, Z = _get_mesh_coordinates(clf, X, y, cols, hist_values)
            ax.contourf(xx,yy,Z, cmap=cm)

def _create_X_grid(X, values, cols):
    """
    Creates a dataframe to be used as input in the classifier for mapping 
    all the values between the minimum and maximum values of a feature. 

    Parameters
    ----------
    X : pandas dataframes
        Set of features 

    values : numpy array
        Either an 1D or 2D array with the values in which the decision boundary
        is going to be mapped

    cols : list
        one or two columns from X.columns that are going to be mapped
        IMPORTANT: when it is 1D, cols should be declared without brackets

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> random_values = np.random.rand(4,4)
    >>> X = pd.DataFrame(random_values, columns=['A','B','C','D'])
    >>> x = [1, 3, 4]
    >>> _create_X_grid(X, x, 'D')
         A    B    C  D
    0  0.0  0.0  0.0  1
    1  0.0  0.0  0.0  3
    2  0.0  0.0  0.0  4
    >>> y = np.ones((2,5))
    >>> y
    array([[ 1.,  1.,  1.,  1.,  1.],
           [ 1.,  1.,  1.,  1.,  1.]])
    >>> _create_X_grid(X, y, ['B','D'])
         A    B    C    D
    0  0.0  1.0  0.0  1.0
    1  0.0  1.0  0.0  1.0
    2  0.0  1.0  0.0  1.0

    """
    
    n_rows = len(values)
    n_columns = len(X.columns)
    X_grid = pd.DataFrame(np.zeros((n_rows,n_columns)), columns=X.columns)
    X_grid[cols] = values

    return X_grid

def _check_cols_bugs(cols):
    """
    Check the following possible bugs:
    - Check if the pair of features to be plotted in cols is more than 2 (3D 
    is not supported)
    - Check if the pair of features to be plotted contain duplicate values, 
    if yes, cols is redefined as the first item
    - Check if cols is 1D (histogram plot) but was declared as a list, if yes
    cols is redefined to the first item of the list

    """
    # Check the number of dimensions to be plotted is higher than expected
    if not isinstance(cols, str): # Perform checks only if cols is not str
        if len(cols)>2:
            # Later I will replace this to a raise error function
            _raise_error("Maximum number of input features exceeded. 'col' should \
                have either one value (1D) or two (2D)")

        # Check if there's duplicates in cols
        if len(cols)==2:
            if cols[0]==cols[1]:
                cols=cols[0] # Redefine cols as 1D (There's an issue when using 
                             # set(cols)

        # Check if cols is declared as a list and convert to a string (issue #1)
        if len(cols)==1:
            if isinstance(cols, list):
                cols=cols[0]

    return cols

def _raise_error(message, type=None):
    print(message)
    pass

def _round_multiple(x, base=5):
    return int(base * np.round(float(x)/base))

def _get_mesh_coordinates(clf, X, y, cols, hist_values=None):
    """
    Takes either one feature or a pair of features and creates a grid 
    ranging 10 steps between their minimum and maximum values and then get the 
    classifier's output for each of these values in the grid (Z value)

    Examples
    --------
    >>> from sklearn.datasets import make_classification
    >>> X, y = make_classification(n_features=4, random_state=42)
    >>> from sklearn.linear_model import LogisticRegression
    >>> clf = LogisticRegression(random_state=42).fit(X,y)
    >>> import pandas as pd
    >>> X = pd.DataFrame(X, columns=['A','B','C','D'])
    >>> xx, yy, Z = get_mesh_pdcoordinates(clf, X, y, 'A')
    >>> import matplotlib.pyplot as plt
    >>> plt.contourf(xx,yy,Z)

    """
    # Check if `cols` contains one string
    if isinstance(cols,str):
        # 1D coordinates
        n_steps = 11 # the value is 11 since there's 11 values in [0,10]
        min_x, max_x = np.min(X[cols]), np.max(X[cols])
        y_width = _round_multiple(1.3*np.max(hist_values[0]), base=5)#max_x - min_x
        x = np.linspace(min_x, max_x, n_steps)
        y = np.linspace(0, y_width, n_steps)

        xx, yy = np.meshgrid(x, y)

        # Create matrix if features that are going to be mapped
        X_grid = _create_X_grid(X, xx.ravel(), cols)

        # Get prediction values (either probabilities or from decision function)
        # if hasattr(clf, "decision_function"):
        #     Z = clf.decision_function(X_grid)
        # else:
        Z = clf.predict(X_grid.values)[:, 0]

        Z = Z.reshape(xx.shape)

        return xx, yy, Z
    # First, check if we will map one or two features (1D or 2D)
    if len(cols) > 2:
        # Later I will replace this to a raise error function
        _raise_error("Maximum number of input features exceeded. 'col' {} should \
            have either one value (1D) or two (2D)".format(cols))

    if len(cols)==2:
        # 2D coordinates
        n_steps = 11 # the value is 11 since there's 11 values in [0,10]
        min_x, max_x = np.min(X[cols[0]]), np.max(X[cols[0]])
        min_y, max_y = np.min(X[cols[1]]), np.max(X[cols[1]])
        x = np.linspace(min_x, max_x, n_steps)
        y = np.linspace(min_y, max_y, n_steps)

        xx, yy = np.meshgrid(x, y)

        # Create matrix if features that are going to be mapped
        X_grid = _create_X_grid(X, np.c_[xx.ravel(), yy.ravel()], cols)

        # Get prediction values (either probabilities or from decision function)
        Z = clf.predict(X_grid.values)[:, 0]

        Z = Z.reshape(xx.shape)

        return xx, yy, Z







    


