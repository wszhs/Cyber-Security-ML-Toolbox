
import copy
import scipy.sparse as sp
import numpy as np

def check_adj(adj):
    """Check if the modified adjacency is symmetric and unweighted.
    """
    assert np.abs(adj - adj.T).sum() == 0, "Input graph is not symmetric"
    assert adj.tocsr().max() == 1, "Max value should be 1!"
    assert adj.tocsr().min() == 0, "Min value should be 0!"

def random_sample_edges( adj, n, exclude):
    itr = sample_forever(adj, exclude=exclude)
    return [next(itr) for _ in range(n)]

def sample_forever( adj, exclude):
    """Randomly random sample edges from adjacency matrix, `exclude` is a set
    which contains the edges we do not want to sample and the ones already sampled
    """
    while True:
        # t = tuple(np.random.randint(0, adj.shape[0], 2))
        # t = tuple(random.sample(range(0, adj.shape[0]), 2))
        t = tuple(np.random.choice(adj.shape[0], 2, replace=False))
        if t not in exclude:
            yield t
            exclude.add(t)
            exclude.add((t[1], t[0]))

def random_attack(features,adj,labels,ids_test,type='add',ptb_rate=0.5):

    n_perturbations=int(ptb_rate * (adj.sum()//2))
    modified_adj=adj.tolil()

    if type == 'flip':
        # sample edges to flip
        edges = random_sample_edges(adj, n_perturbations, exclude=set())
        for n1, n2 in edges:
            modified_adj[n1, n2] = 1 - modified_adj[n1, n2]
            modified_adj[n2, n1] = 1 - modified_adj[n2, n1]

    if type == 'add':
        # sample edges to add
        nonzero = set(zip(*adj.nonzero()))
        edges = random_sample_edges(adj, n_perturbations, exclude=nonzero)
        for n1, n2 in edges:
            modified_adj[n1, n2] = 1
            modified_adj[n2, n1] = 1

    if type == 'remove':
        # sample edges to remove
        nonzero = np.array(sp.triu(adj, k=1).nonzero()).T
        indices = np.random.permutation(nonzero)[: n_perturbations].T
        modified_adj[indices[0], indices[1]] = 0
        modified_adj[indices[1], indices[0]] = 0

    # check_adj(modified_adj)
    return modified_adj