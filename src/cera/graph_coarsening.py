"""Coarsening-aware agreement between related graph partitions.

The procedure does not access ground-truth labels.
"""
import numpy as np
from sklearn.metrics import adjusted_rand_score
from .graph_mst import mst_gap_labels,kcount
from .density_base import predict_density_base

def predict_graph_coarsening(X, *, return_info=False):
    X=np.asarray(X,float)
    if X.ndim!=2 or len(X)<33 or not np.isfinite(X).all():
        raise ValueError('X must have finite (n,p) with n>=33')
    a,_=mst_gap_labels(X,k=12,threshold=.85,min_frac=.05)
    k12=kcount(a)
    b=None
    k16=0
    agreement=float('nan')
    branch='density_fallback'
    if k12>=2:
        b,_=mst_gap_labels(X,k=16,threshold=.85,min_frac=.05)
        k16=kcount(b)
        agreement=float(adjusted_rand_score(a,b))
        if k16==k12 and agreement>.85:
            labels=a
            branch='stable_sameK'
        elif k12>=3 and k16>=2 and k12>k16 and agreement>.50:
            count=np.bincount(a[a>=0]);small=float(count.min()/len(a))
            # When fine split has small pieces, favor supported coarse labeling.
            # Otherwise retain more resolved fine structure.
            labels=b if small<.15 else a
            branch='coarsening_coarse' if small<.15 else 'coarsening_fine'
        else:
            labels=predict_density_base(X)
    else:
        labels=predict_density_base(X)
    info={'branch':branch,'K12':k12,'K16':k16,
          'agreement':agreement,'K_pred':kcount(labels),
          'coverage':float(np.mean(labels>=0))}
    return (labels,info) if return_info else labels
