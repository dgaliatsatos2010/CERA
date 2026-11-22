"""Projection screening using PCA and Gaussian mixture criteria.

Heuristic thresholds; no type-I error guarantee.
"""
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from .graph_coarsening import predict_graph_coarsening
from .graph_mst import kcount

PARAMS = {'min_dim':5, 'delta_bic':30.0, 'min_sep':2.8, 'min_weight':0.12,
          'max_valley_ratio':0.80, 'seed':1720}

def _gauss_density(x, mu, sigma):
    return np.exp(-.5*((x-mu)/sigma)**2)/(np.sqrt(2*np.pi)*sigma)

def projection_candidate(X):
    """Train-only unsupervised GMM gate on first PCA score; no labels or family info.
    Returns labels if two clearly separated density modes, otherwise None.
    """
    X=np.asarray(X,float)
    n,p=X.shape
    if p<PARAMS['min_dim'] or n<120:return None, {'reason':'ineligible_dimension'}
    Z=X-X.mean(axis=0,keepdims=True)  # raw-unit PCA; stress-test sensitivity to feature units
    scores=PCA(n_components=1,svd_solver='full').fit_transform(Z)
    g1=GaussianMixture(n_components=1,n_init=2,random_state=PARAMS['seed'],reg_covar=1e-5).fit(scores)
    g2=GaussianMixture(n_components=2,n_init=3,random_state=PARAMS['seed'],reg_covar=1e-5).fit(scores)
    delta=float(g1.bic(scores)-g2.bic(scores))
    inds=np.argsort(g2.means_.ravel())
    mu=g2.means_.ravel()[inds]; sig=np.sqrt(g2.covariances_.ravel()[inds]); weights=g2.weights_[inds]
    sep=float((mu[1]-mu[0])/np.sqrt(np.mean(sig**2)))
    if not (delta>PARAMS['delta_bic'] and sep>PARAMS['min_sep'] and weights.min()>PARAMS['min_weight']):
        return None,{'reason':'weak_evidence','delta_bic':delta,'separation':sep,'min_weight':float(weights.min())}
    midpoint=(mu[0]+mu[1])/2
    valley=float(sum(weights*_gauss_density(midpoint,mu,sig)))
    peaks=np.array([sum(weights*_gauss_density(m,mu,sig)) for m in mu])
    ratio=float(valley/max(1e-12,peaks.min()))
    if ratio>PARAMS['max_valley_ratio']:
        return None,{'reason':'no_deep_valley','ratio':ratio}
    labels=g2.predict(scores).astype(int)
    if min(np.bincount(labels)) < max(12,int(.05*n)):
        return None,{'reason':'small_predicted_cluster'}
    return labels,{'reason':'accepted','delta_bic':delta,'separation':sep,'valley_ratio':ratio}

def predict_projection_gate(X,return_info=False):
    X=np.asarray(X,float)
    if X.ndim!=2 or len(X)<33 or not np.isfinite(X).all():
        raise ValueError('X finite (n,p) and n>=33 required')
    previous=predict_graph_coarsening(X)
    candidate,info=projection_candidate(X)
    # We only override a single group; an existing successful multi-group split is conserved.
    out=candidate if candidate is not None and kcount(previous)<=1 else previous
    metadata={'branch':'projection_2mode' if out is candidate else 'graph_base', **info,
              'K_pred':int(kcount(out))}
    return (out,metadata) if return_info else out
