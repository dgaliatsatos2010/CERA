"""Robust unit canonicalization and selective projection screening.

Positive feature-wise affine invariance is not claimed as a new property.
"""
import numpy as np
from scipy.stats import kurtosis
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import RobustScaler
from .graph_coarsening import predict_graph_coarsening
from .graph_mst import kcount
from .projection_gate import _gauss_density

PARAMS = dict(min_dim=5, max_coordinates=6, delta_bic=36.0,
              min_sep=2.8, min_weight=.12, max_valley_ratio=.75, seed=1725)

def canonicalize(X):
    X=np.asarray(X,dtype=float)
    if X.ndim!=2 or len(X)<33 or not np.isfinite(X).all():
        raise ValueError('finite (n,p) data with n>=33 required')
    # Affine equivariance for positive changes in measurement units, except near-degenerate columns.
    return RobustScaler(quantile_range=(15,85),unit_variance=False).fit_transform(X)

def one_d_gate(scores, n_features=1):
    x=np.asarray(scores,dtype=float).reshape(-1,1)
    if float(np.std(x))<1e-9:return None, {'reason':'constant'}
    g1=GaussianMixture(n_components=1,n_init=1,random_state=PARAMS['seed'],reg_covar=1e-5).fit(x)
    g2=GaussianMixture(n_components=2,n_init=2,random_state=PARAMS['seed'],reg_covar=1e-5).fit(x)
    delta=float(g1.bic(x)-g2.bic(x))
    inds=np.argsort(g2.means_.ravel())
    mu=g2.means_.ravel()[inds];sig=np.sqrt(g2.covariances_.ravel()[inds]);weights=g2.weights_[inds]
    sep=float((mu[1]-mu[0])/np.sqrt(np.mean(sig**2)))
    midpoint=(mu[0]+mu[1])/2
    valley=float(np.sum(weights*_gauss_density(midpoint,mu,sig)))
    peaks=np.array([np.sum(weights*_gauss_density(m,mu,sig)) for m in mu])
    ratio=float(valley/max(1e-12,peaks.min()))
    # Bonferroni-like screening penalty; *not* an FWER-valid test.
    bound=PARAMS['delta_bic']+2*np.log(max(2,n_features))
    if not(delta>bound and sep>PARAMS['min_sep'] and weights.min()>PARAMS['min_weight'] and ratio<PARAMS['max_valley_ratio']):
        return None, dict(reason='weak_evidence',bic_gain=delta,separation=sep,valley_ratio=ratio)
    y=g2.predict(x).astype(int)
    if np.min(np.bincount(y)) < max(12,int(.05*len(y))):
        return None, {'reason':'small_cluster','bic_gain':delta}
    return y, {'reason':'accepted','bic_gain':delta,'separation':sep,'valley_ratio':ratio}

def predict_scale_screen(X, return_info=False):
    Z=canonicalize(X);n,p=Z.shape
    previous=predict_graph_coarsening(Z)
    if kcount(previous)>1 or p<PARAMS['min_dim']:
        res=previous;detail={'branch':'graph_scaled_base','K_pred':kcount(res)}
        return (res,detail) if return_info else res
    # Feature shortlist is label-free and itself invariant to positive feature-unit changes.
    K=kurtosis(Z,axis=0,fisher=True,bias=False,nan_policy='omit')
    K=np.nan_to_num(K,nan=1e6,posinf=1e6,neginf=-1e6)
    shortlist=sorted(range(p),key=lambda j:(float(K[j]),j))[:PARAMS['max_coordinates']]
    scores=[(f'coordinate_{j}',Z[:,j]) for j in shortlist]
    scores.append(('PCA1_scaled',PCA(n_components=1,svd_solver='full').fit_transform(Z).ravel()))
    candidates=[]
    for name,s in scores:
        lab,info=one_d_gate(s,len(scores))
        if lab is not None: candidates.append((info['bic_gain'],name,lab,info))
    if candidates:
        _,name,res,info=max(candidates,key=lambda a:a[0]);detail={'branch':name,'K_pred':kcount(res),**info}
    else:
        res=previous;detail={'branch':'graph_scaled_base','K_pred':kcount(res)}
    return (res,detail) if return_info else res

def predict_projection_scaled(X):
    from .projection_gate import predict_projection_gate
    return predict_projection_gate(canonicalize(X))
