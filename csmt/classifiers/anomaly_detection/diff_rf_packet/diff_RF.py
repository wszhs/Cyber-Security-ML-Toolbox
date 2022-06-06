#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 12:19:32 2020

@author: Pierre-François Marteau (https://people.irisa.fr/Pierre-Francois.Marteau/)
"""

# Inspired from an implementation of the isolation forest algorithm provided at
# https://github.com/xhan0909/isolation_forest

import numpy as np
import time
from functools import partial
from multiprocessing import Pool

import random as rn

def getSplit(X):
    """
    Randomly selects a split value from set of scalar data 'X'.
    Returns the split value.
    
    Parameters
    ----------
    X : array 
        Array of scalar values
    Returns
    -------
    float
        split value
    """
    xmin = X.min()
    xmax = X.max()
    return np.random.uniform(xmin, xmax)

def similarityScore(S, node, alpha):
    """
    Given a set of instances S falling into node and a value alpha >=0,
    returns for all element x in S the weighted similarity score between x
    and the centroid M of S (node.M)
    
    Parameters
    ----------
    S : array  of instances
        Array  of instances that fall into a node
    node: a DiFF tree node
        S is the set of instances "falling" into the node
    alpha: float
        alpha is the distance scaling hyper-parameter
    Returns
    -------
    array
        the array of similarity values between the instances in S and the mean of training instances falling in node

    """
    d = np.shape(S)[1]
    if len(S) > 0:
        d = np.shape(S)[1]
        U = (S-node.M)/node.Mstd # normalize using the standard deviation vector to the mean
        U = (2)**(-alpha*(np.sum(U*U/d, axis=1)))
    else:
        U = 0

    return U


def EE(hist):
    """
    given a list of positive values as a histogram drawn from any information source,
    returns the empirical entropy of its discrete probability function.
    
    Parameters
    ----------
    hist: array 
        histogram
    Returns
    -------
    float
        empirical entropy estimated from the histogram

    """
    h = np.asarray(hist, dtype=np.float64)
    if h.sum() <= 0 or (h < 0).any():
        return 0
    h = h/h.sum()
    return -(h*np.ma.log2(h)).sum()


def weightFeature(s, nbins):
    '''
    Given a list of values corresponding to a feature dimension, returns a weight (in [0,1]) that is 
    one minus the normalized empirical entropy, a way to characterize the importance of the feature dimension. 
    
    Parameters
    ----------
    s: array 
        list of scalar values corresponding to a feature dimension
    nbins: int
        the number of bins used to discretize the feature dimension using an histogram.
    Returns
    -------
    float
        the importance weight for feature s.
    '''
    wmin=.02
    # if not np.isfinite(mins) or not np.isfinite(maxs) or np.abs(mins- maxs)<1e-300:
    #     return 1e-4

    hist, bin_edges = np.histogram(s, bins=nbins)
    #hist = histogram1d(s, range=[mins-1e-4,maxs+1e-4], bins=nbins)
    ent = EE(hist)
    ent = ent/np.log2(nbins)
    if np.isfinite(ent):
         #return max(1/2-abs(1/2-ent), wmin)
         return max(1-ent, wmin)
    else:
         return wmin


def walk_tree(forest, node, treeIdx, obsIdx, X, featureDistrib, depth=0, alpha=1e-2):
    '''
    Recursive function that walks a tree from an already fitted forest to compute the path length
    of the new observations.
    
    Parameters
    ----------
    forest : DiFF_RF 
        A fitted forest of DiFF trees
    node: DiFF Tree node
        the current node
    treeIdx: int
        index of the tree that is being walked.
    obsIdx: array
        1D array of length n_obs. 1/0 if the obs has reached / has not reached the node.
    X: nD array. 
        array of observations/instances.
    depth: int
        current depth.
    Returns
    -------
    None
    '''

    if isinstance(node, LeafNode):
        Xnode = X[obsIdx]
        f = ((node.size+1)/forest.sample_size) / ((1+len(Xnode))/forest.XtestSize)
        if alpha == 0:
            forest.LD[obsIdx, treeIdx] = 0
            forest.LF[obsIdx, treeIdx] = -f
            forest.LDF[obsIdx, treeIdx] = -f
        else:
            z = similarityScore(Xnode, node, alpha)
            forest.LD[obsIdx, treeIdx] = z
            forest.LF[obsIdx, treeIdx] = -f
            forest.LDF[obsIdx, treeIdx] = z*f

    else:

        idx = (X[:, node.splitAtt] <= node.splitValue) * obsIdx
        walk_tree(forest, node.left, treeIdx, idx, X, featureDistrib, depth + 1, alpha=alpha)

        idx = (X[:, node.splitAtt] > node.splitValue) * obsIdx
        walk_tree(forest, node.right, treeIdx, idx, X, featureDistrib, depth + 1, alpha=alpha)


def create_tree(X, featureDistrib, sample_size, max_height):
    '''
    Creates an DiFF tree using a sample of size sample_size of the original data.
        
    Parameters
    ----------
    X: nD array. 
        nD array with the observations. Dimensions should be (n_obs, n_features).
    sample_size: int
        Size of the sample from which a DiFF tree is built.
    max_height: int
        Maximum height of the tree.
    Returns
    -------
    a DiFF tree
    '''
    rows = np.random.choice(len(X), sample_size, replace=False)
    featureDistrib = np.array(featureDistrib)
    return DiFF_Tree(max_height).fit(X[rows, :], featureDistrib)


class DiFF_TreeEnsemble:
    '''
    DiFF Forest.
    Even though all the methods are thought to be public the main functionality of the class is given by:
    - __init__
    - __fit__
    - __predict__
    '''
    def __init__(self, n_trees: int = 10):
        '''
        Creates the DiFF-RF object.
        
        Parameters
        ----------
        sample_size: int. 
            size of the sample randomly drawn from the train instances to build each DiFF tree.  
        n_trees: int
            The number of trees in the forest
        Returns
        -------
            None
        '''

        self.n_trees = n_trees
        self.alpha=1.0
        np.random.seed(int(time.time()))
        rn.seed(int(time.time()))


    def fit(self, X: (np.ndarray), n_jobs: int = 8):
        """
        Fits the algorithm into a model.
        Given a 2D matrix of observations, create an ensemble of IsolationTree
        objects and store them in a list: self.trees.  Convert DataFrames to
        ndarray objects.
        Uses parallel computing.
        
        Parameters
        ----------
        X: nD array. 
            nD array with the train instances. Dimensions should be (n_obs, n_features).  
        n_jobs: int
            number of parallel jobs that will be launched
        Returns
        -------
            the object itself.
        """
        self.X = X
        self.path_normFactor = np.sqrt(len(X))

        # self.sample_size = min(self.sample_size, 0.5*len(X))
        self.sample_size = int(0.33*len(X))

        limit_height = 1.0*np.ceil(np.log2(self.sample_size))

        featureDistrib = []
        nbins = int(len(X)/8)+2
        for i in range(np.shape(X)[1]):
            featureDistrib.append(weightFeature(X[:, i], nbins))
        featureDistrib = np.array(featureDistrib)
        featureDistrib = featureDistrib/(np.sum(featureDistrib)+1e-5)
        self.featureDistrib = featureDistrib

        create_tree_partial = partial(create_tree,
                                      featureDistrib=self.featureDistrib,
                                      sample_size=self.sample_size,
                                      max_height=limit_height)

        with Pool(n_jobs) as p:
            self.trees = p.map(create_tree_partial,
                               [X for _ in range(self.n_trees)]
                               )
        return self


    def walk(self, X: np.ndarray) -> np.ndarray:
        """
        Given a nD matrix of observations, X, compute the average path length,
        the distance, frequency and collective anomaly scores
        for instances in X.  Compute the path length for x_i using every
        tree in self.trees then compute the average for each x_i.  Return an
        ndarray of shape (len(X),1).
        
        Parameters
        ----------
        X: nD array. 
            nD array with the instances to be tested. Dimensions should be (n_obs, n_features).   
        Returns
        -------
            None
        """

        self.L = np.zeros((len(X), self.n_trees))
        self.LD = np.zeros((len(X), self.n_trees))
        self.LF = np.zeros((len(X), self.n_trees))
        self.LDF = np.zeros((len(X), self.n_trees))

        for treeIdx, itree in enumerate(self.trees):
            obsIdx = np.ones(len(X)).astype(bool)
            walk_tree(self, itree, treeIdx, obsIdx, X, self.featureDistrib, alpha=self.alpha)


    def decision_function(self, X: np.ndarray, alpha=0.1) -> np.ndarray:
        """
        Given a nD matrix of observations, X, compute the anomaly scores
        for instances in X, returning 3 1D arrays of anomaly scores
        
        Parameters
        ----------
        X: nD array. 
            nD array with the tested observations to be predicted. Dimensions should be (n_obs, n_features).   
        alpha: float
            scaling distance hyper-parameter.
        Returns
        -------
        scD, scF, scFF: 1d arrays
            respectively the distance scores (point-wise anomaly score), the frequency of visit socres and the collective anomaly scores
        """
        self.XtestSize = len(X)
        self.alpha = alpha

        # Evaluate the scores for each of the observations.
        self.walk(X)

        scDF =self.LDF.mean(1)

        return scDF
    

    def predict_from_anomaly_scores(self, scores: np.ndarray, threshold: float) -> np.ndarray:
        """
        Given an array of scores and a score threshold, return an array of
        the predictions: 1 for any score >= the threshold and 0 otherwise.
        
        Parameters
        ----------
        scores: 1D array. 
            1D array of scores. Dimensions should be (n_obs, n_features).   
        threshold: float
            Threshold for considering a observation an anomaly, the higher the less anomalies.
        Returns
        -------
        1D array
            The prediction array corresponding to 1/0 if anomaly/not anomaly respectively.

        :param scores: 1D array. Scores produced by the random forest.
        :param threshold: Threshold for considering a observation an anomaly, the higher the less anomalies.
        :return: Return predictions
        """
        out = scores >= threshold
        return out*1
    

    def predict(self, X: np.ndarray, threshold: float) -> np.ndarray:
        """
        A shorthand for calling anomaly_score() and predict_from_anomaly_scores().
        
        Parameters
        ----------
        X: nD array. 
            nD array with the tested observations to be predicted. Dimensions should be (n_obs, n_features).   
        threshold: float
            Threshold for considering a observation an anomaly, the higher the less anomalies.
        Returns
        -------
        1D array
            The prediction array corresponding to 1/0 if anomaly/not anomaly respectively.
        """

        scores = self.anomaly_score(X)
        return self.predict_from_anomaly_scores(scores, threshold)


class DiFF_Tree:
    '''
    Construct a tree via randomized splits with maximum height height_limit.
    '''
    def __init__(self, height_limit):
        '''
        Parameters
        ----------
        height_limit: int
            Maximum height of the tree.
        Returns
        -------
        None
        '''
        self.height_limit = height_limit

    def fit(self, X: np.ndarray, featureDistrib: np.array):
        """
        Given a 2D matrix of observations, create an DiFF tree. Set field
        self.root to the root of that tree and return it.
        
        Parameters
        ----------
        X: nD array. 
            nD array with the observations. Dimensions should be (n_obs, n_features).        
        featureDistrib: 1D array
            The distribution weight affected to each dimension
        Returns
        -------
        A DIFF tree root.
        """
        self.root = InNode(X, self.height_limit, featureDistrib, len(X), 0)

        return self.root


class InNode:
    '''
    Node of the tree that is not a leaf node.
    The functionality of the class is:
    - Do the best split from a sample of randomly chosen
        dimensions and split points.
    - Partition the space of observations according to the
    split and send the along to two different nodes
    The method usually has a higher complexity than doing it for every point.
    But because it's using NumPy it's more efficient time-wise.
    '''
    def __init__(self, X, height_limit, featureDistrib, sample_size, current_height):
        '''
        Parameters
        ----------
        X: nD array. 
            nD array with the training instances that have reached the node.
        height_limit: int
            Maximum height of the tree.
        Xf: nD array. 
            distribution used to randomly select a dimension (feature) used at parent level. 
        sample_size: int
            Size of the sample used to build the tree.
        current_height: int
            Current height of the tree.
        Returns
        -------
            None
        '''

        self.size = len(X)
        self.height = current_height+1
        n_obs, n_features = X.shape
        next_height = current_height + 1
        limit_not_reached = height_limit > next_height

        if len(X) > 32:
            featureDistrib = []
            nbins = int(len(X)/8)+2
            for i in range(np.shape(X)[1]):
                featureDistrib.append(weightFeature(X[:, i], nbins))
            featureDistrib = np.array(featureDistrib)
            featureDistrib = featureDistrib/(np.sum(featureDistrib)+1e-5)

        self.featureDistrib = featureDistrib

        cols = np.arange(np.shape(X)[1], dtype='int')

        self.splitAtt = rn.choices(cols, weights=featureDistrib)[0]
        splittingCol = X[:, self.splitAtt]
        self.splitValue = getSplit(splittingCol)
        idx = splittingCol <= self.splitValue

        idx = splittingCol <= self.splitValue

        X_aux = X[idx, :]

        self.left = (InNode(X_aux, height_limit, featureDistrib, sample_size, next_height)
                     if limit_not_reached and X_aux.shape[0] > 5 and (np.any(X_aux.max(0) != X_aux.min(0))) else LeafNode(
                         X_aux, next_height, X, sample_size))

        idx = np.invert(idx)
        X_aux = X[idx, :]
        self.right = (InNode(X_aux, height_limit, featureDistrib, sample_size, next_height)
                      if limit_not_reached and X_aux.shape[0] > 5 and (np.any(X_aux.max(0) != X_aux.min(0))) else LeafNode(
                          X_aux, next_height, X, sample_size))

        self.n_nodes = 1 + self.left.n_nodes + self.right.n_nodes


class LeafNode:
    '''
    Leaf node
    The base funcitonality is storing the Mean and standard deviation of the observations in that node.
    We also evaluate the frequency of visit for training data.
    '''
    def __init__(self, X, height, Xp, sample_size):
        '''
        Parameters
        ----------
        X: nD array. 
            nD array with the training instances falling into the leaf node.    
        height: int
            Current height of the tree.
        Xf: nD array. 
            nD array with the training instances falling into the parent node.    
        sample_size: int
            Size of the sample used to build the tree.
        Returns
        -------
            None
        '''
        self.height = height+1
        self.size = len(X)
        self.n_nodes = 1
        self.freq = self.size/sample_size
        self.freqs = 0

        if len(X) != 0:
            self.M = np.mean(X, axis=0)
            if len(X) > 10:
                self.Mstd = np.std(X, axis=0)
                self.Mstd[self.Mstd == 0] = 1e-2
            else:
                self.Mstd = np.ones(np.shape(X)[1])
        else:
            self.M = np.mean(Xp, axis=0)
            if len(Xp) > 10:
                self.Mstd = np.std(Xp, axis=0)
                self.Mstd[self.Mstd == 0] = 1e-2
            else:
                self.Mstd = np.ones(np.shape(X)[1])

