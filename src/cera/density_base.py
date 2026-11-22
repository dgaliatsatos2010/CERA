"""Conservative density clustering and reassignment of labeled noise points.

Heuristic research implementation; no population-topology guarantee.
"""
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import HDBSCAN, DBSCAN, OPTICS
from sklearn.metrics import adjusted_rand_score

def kcount(labels):
    return int(len(set(labels) - {-1}))

def predict_hdbscan(X):
    return HDBSCAN(min_cluster_size=max(30, int(.045*len(X))), min_samples=15,
                   allow_single_cluster=True, n_jobs=1).fit_predict(X)

def predict_dbscan_auto(X):
    dist=NearestNeighbors(n_neighbors=16,n_jobs=1).fit(X).kneighbors(X,return_distance=True)[0][:,-1]
    return DBSCAN(eps=1.5*float(np.quantile(dist,.8)), min_samples=15,n_jobs=1).fit_predict(X)

def predict_density_consensus(X):
    dist=NearestNeighbors(n_neighbors=16,n_jobs=1).fit(X).kneighbors(X,return_distance=True)[0][:,-1]
    q80=float(np.quantile(dist,.8))
    eps=np.array([.85,1.05,1.3,1.65,2.05,2.5,3.0])*q80
    labs=[DBSCAN(eps=float(e),min_samples=15,n_jobs=1).fit_predict(X) for e in eps]
    ks=np.array([kcount(l) for l in labs]);cov=np.array([np.mean(l>=0) for l in labs])
    cand=[]
    for i in range(1,len(labs)-1):
        if not (2<=ks[i]<=6 and ks[i-1]==ks[i]==ks[i+1] and min(cov[i-1:i+2])>=.85):continue
        sim=min(adjusted_rand_score(labs[i-1],labs[i]),adjusted_rand_score(labs[i],labs[i+1]))
        if sim < .88:continue
        cand.append((sim+.1*cov[i],i))
    if cand:return labs[max(cand)[1]], 'DBSCAN_plateau'
    return predict_hdbscan(X),'HDBSCAN_fallback'

def recover_noise(X, base, *, scale_factor=1.40, margin_ratio=.70, k=15):
    """Only reassign previous -1 points; never change an existing non-noise label.
    For cluster c, scale is 90th percentile within-cluster kNN distance.
    Point may be assigned to nearest cluster only if closest labeled member
    is within scale_factor * scale and (if K>1) next/nearest distance margin.
    """
    X=np.asarray(X,dtype=float)
    out=np.asarray(base,dtype=int).copy()
    classes=np.unique(out[out>=0]); noise=np.flatnonzero(out<0)
    if len(classes)==0 or len(noise)==0: return out
    dists=np.zeros((len(noise),len(classes)))
    scales=[]
    for j,cl in enumerate(classes):
        Z=X[out==cl]; kn=min(k+1,len(Z))
        if kn<3:scales.append(0.);dists[:,j]=np.inf;continue
        near=NearestNeighbors(n_neighbors=kn,n_jobs=1).fit(Z)
        within=near.kneighbors(Z,return_distance=True)[0][:,-1]
        scales.append(float(np.quantile(within,.90)))
        dists[:,j]=NearestNeighbors(n_neighbors=1,n_jobs=1).fit(Z).kneighbors(X[noise],return_distance=True)[0][:,0]
    ind=np.argmin(dists,axis=1)
    d1=dists[np.arange(len(noise)),ind]
    s=np.asarray(scales)[ind]
    acc=d1 <= scale_factor*s
    if len(classes)>1:
        d2=np.partition(dists,1,axis=1)[:,1]
        acc &= (d1<=margin_ratio*d2)
    out[noise[acc]]=classes[ind[acc]]
    return out

def predict_density_base(X):
    base=predict_hdbscan(X)
    return recover_noise(X,base)

def predict_optics(X):
    return OPTICS(min_samples=15,min_cluster_size=max(30,int(.045*len(X))),cluster_method='xi', xi=.05, n_jobs=1).fit_predict(X)
