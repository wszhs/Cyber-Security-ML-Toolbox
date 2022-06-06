from sklearn.manifold import TSNE,Isomap,MDS,SpectralEmbedding
from sklearn.decomposition import PCA,TruncatedSVD,IncrementalPCA,SparsePCA,FactorAnalysis,FastICA,DictionaryLearning

def tsne_dim_redu(X,n_components):
    tsne = TSNE(n_components=n_components) 
    X = tsne.fit_transform(X) 
    return X

def pca_dim_redu(X,n_components):
    pca = PCA(n_components=n_components)
    X = pca.fit_transform(X)
    return X

def svd_dim_redu(X,n_components):
    svd = TruncatedSVD(n_components = n_components)
    X = svd.fit_transform(X)
    return X