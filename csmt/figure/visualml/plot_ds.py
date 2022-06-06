import matplotlib
matplotlib.use('TKAgg')
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

def plot_ds_2d(X,y):
    Xy_df = pd.DataFrame(data = X, columns = ['comp0', 'comp1'])
    Xy_df['label'] = y

    # Set style of scatterplot
    sns.set_context("notebook", font_scale=1.1)
    sns.set_style("ticks")

    # Create scatterplot of dataframe
    sns.lmplot(x='comp0',
            y='comp1',
            data=Xy_df,
            fit_reg=False,
            legend=True,
            height=9,
            hue = 'label',
            scatter_kws={"s":200, "alpha":0.3})

    plt.title('Results:', weight='bold').set_fontsize('14')
    plt.xlabel('Comp 0', weight='bold').set_fontsize('10')
    plt.ylabel('Comp 1', weight='bold').set_fontsize('10')
    plt.show()
    
def plot_ds_3d(X,y):
    Xy_df = pd.DataFrame(data = X, columns = ['comp0', 'comp1','comp2'])
    Xy_df['label'] = y
    fig=plt.figure(figsize = (10, 10))
    ax=Axes3D(fig)
    # ax = plt.axes(projection='3d')
    ax.scatter(
        xs=Xy_df.loc[:, 'comp0'], 
        ys=Xy_df.loc[:, 'comp1'], 
        zs=Xy_df.loc[:, 'comp2'], 
        c=Xy_df.loc[:, 'label'], 
        cmap=plt.get_cmap('bwr')
    )
    ax.set_xlabel('comp 0')
    ax.set_ylabel('comp 1')
    ax.set_zlabel('comp 2')
    ax.legend()
    plt.title('Results:', weight='bold').set_fontsize('14')
    plt.show()


