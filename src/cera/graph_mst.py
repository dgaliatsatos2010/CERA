"""Graph separation using normalized k-nearest-neighbor minimum spanning trees.

Heuristic research implementation, not a calibrated hypothesis test.
"""
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.cluster.hierarchy import fcluster
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import HDBSCAN
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import pairwise_distances


def kcount(y): return len(set(y)-{-1})


def mst_gap_labels(X, k=12, threshold=2.0, min_frac=.05, max_clusters=6):
    """Split along globally anomalously long MST edges relative to local kNN scale.
    Components not reaching min_cluster are assigned to noise. Avoids any truth labels.
    """
    X=np.asarray(X,float); n=len(X)
    if n<2*k+1: return np.zeros(n,dtype=int), {'reason':'n_small','max_stress':0.}
    # Scale all dimensions with robust scale; preserve relative density inside a dimension.
    Z=RobustScaler(quantile_range=(15,85)).fit_transform(X)
    nbr=NearestNeighbors(n_neighbors=min(n,k+1),n_jobs=1).fit(Z)
    distances,inds=nbr.kneighbors(Z)
    core=distances[:,-1].clip(min=1e-10)
    # build 12-NN graph, weights are metric distances, normalized for local density.
    rows=np.repeat(np.arange(n),k);cols=inds[:,1:k+1].flatten()
    raw=distances[:,1:k+1].flatten()
    weight=raw / np.sqrt(core[rows]*core[cols])
    # MST of the normalized graph (a forest if the kNN graph is disconnected).
    directional=coo_matrix((weight,(rows,cols)),shape=(n,n)).tocsr()
    graph=directional.maximum(directional.T)
    mst=minimum_spanning_tree(graph).tocoo()
    # Descending MST edges: cut only prominent bottlenecks with sufficiently large sides.
    edges=sorted(zip(mst.data,mst.row,mst.col),reverse=True)
    parent=np.arange(n)
    def find(a):
        while parent[a]!=a:
            parent[a]=parent[parent[a]];a=parent[a]
        return a
    # Delete all edges with stress > threshold and reconstruct forest.
    for val,i,j in edges:
        if val <= threshold:
            ri,rj=find(i),find(j)
            if ri!=rj:parent[ri]=rj
    roots=np.array([find(i) for i in range(n)])
    _,ids,counts=np.unique(roots,return_inverse=True,return_counts=True)
    minimum=max(12,int(np.ceil(min_frac*n)))
    large=np.flatnonzero(counts>=minimum)
    if not(2<=len(large)<=max_clusters):
        return np.zeros(n,dtype=int), {'reason':'no_split','max_stress':float(max(mst.data)) if len(mst.data) else 0., 'large_k':len(large)}
    label=np.full(n,-1,dtype=int)
    for i,cl in enumerate(large):label[ids==cl]=i
    return label, {'reason':'split','max_stress':float(max(mst.data)) if len(mst.data) else 0., 'large_k':len(large)}


def predict_graph_mst(X, k=12, threshold=2., min_frac=.05):
    return mst_gap_labels(X,k=k,threshold=threshold,min_frac=min_frac)[0]
