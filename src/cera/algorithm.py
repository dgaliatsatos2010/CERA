"""CERA: holdout-checked projection screening for minority-cluster recovery.

CERA is an exploratory clustering method using heuristic thresholds.
Sample splitting does not by itself confer controlled type-I error.
No training label or dataset-family identifier is used in prediction.
"""
import numpy as np
from scipy.stats import kurtosis
from scipy.special import logsumexp
from sklearn.mixture import GaussianMixture
from .scale_screen import canonicalize, predict_scale_screen
from .graph_mst import kcount

SEED=7619
PARAMS=dict(max_candidates=8,min_p=5,train_fraction=.65,
            bic2=18.,bic3=24.,sep2=3.0,sep3=2.6,
            min_prop2=.025,min_prop3=.060,peak_ratio=.67,
            valid_count2=3,valid_count3=4)

def _density(v, mu, sigma, weight):
    return np.sum(weight*np.exp(-.5*((v[...,None]-mu)/sigma)**2)/(np.sqrt(2*np.pi)*sigma),axis=-1)

def _test_gmm_1d(train,valid,n_components):
    a=np.asarray(train,float).reshape(-1,1)
    g1=GaussianMixture(1,random_state=SEED,n_init=1,reg_covar=1e-4).fit(a)
    gm=GaussianMixture(n_components,random_state=SEED,n_init=2,reg_covar=1e-4).fit(a)
    bic_gain=float(g1.bic(a)-gm.bic(a))
    if bic_gain < (PARAMS['bic2'] if n_components==2 else PARAMS['bic3']):return None
    order=np.argsort(gm.means_.ravel());mu=gm.means_.ravel()[order]
    sigma=np.sqrt(gm.covariances_.reshape(-1)[order]);weight=gm.weights_[order]
    threshold=(PARAMS['min_prop2'] if n_components==2 else PARAMS['min_prop3'])
    if weight.min()<threshold:return None
    sep=np.diff(mu)/np.sqrt((sigma[1:]**2+sigma[:-1]**2)/2)
    if np.min(sep)<(PARAMS['sep2'] if n_components==2 else PARAMS['sep3']):return None
    if n_components==3:
        gm2=GaussianMixture(2,random_state=SEED,n_init=1,reg_covar=1e-4).fit(a)
        if gm2.bic(a)-gm.bic(a)<14:return None
    boundaries=[];ratios=[]
    for i in range(n_components-1):
        grid=np.linspace(mu[i],mu[i+1],81)
        dens=_density(grid,mu,sigma,weight)
        # valley can only lie strictly between adjacent fitted means
        ix=1+np.argmin(dens[1:-1]);cut=float(grid[ix]);boundaries.append(cut)
        ratio=float(dens[ix]/min(_density(np.array([mu[i]]),mu,sigma,weight)[0],
                                       _density(np.array([mu[i+1]]),mu,sigma,weight)[0]))
        ratios.append(ratio)
        if ratio>PARAMS['peak_ratio']:return None
    # Split-validation: independent observations must populate all proposed components
    train_group=np.searchsorted(boundaries,np.asarray(train),side='right')
    valid_group=np.searchsorted(boundaries,np.asarray(valid),side='right')
    ct=np.bincount(train_group,minlength=n_components)
    cv=np.bincount(valid_group,minlength=n_components)
    if ct.min()<max(5,int(.022*len(train))):return None
    if cv.min()<(PARAMS['valid_count2'] if n_components==2 else PARAMS['valid_count3']):return None
    pt=ct/len(train);pv=cv/len(valid)
    if np.any(abs(pt-pv)>np.maximum(.085,1.4*pt)):return None
    # Held-out valley evidence: core intervals near each mode must dominate cuts
    # independent from the model; wide gaps measured on validation subset.
    widths=np.minimum(sigma[:-1],sigma[1:])*0.85
    for i,cut in enumerate(boundaries):
        w=float(max(.15,widths[i]))
        valley=np.mean(abs(valid-cut)<=w)
        core1=np.mean(abs(valid-mu[i])<=w)
        core2=np.mean(abs(valid-mu[i+1])<=w)
        if min(core1,core2)==0 or valley>PARAMS['peak_ratio']*min(core1,core2):return None
    return {'K':n_components,'boundaries':boundaries,'bic_gain':bic_gain,'sep':float(np.min(sep)),
            'min_weight':float(np.min(weight)),'valley_ratio':float(max(ratios))}

def rescue_candidate(Z):
    n,p=Z.shape
    if p<PARAMS['min_p'] or n<180:return None, {'reason':'ineligible'}
    rng=np.random.default_rng(SEED+n*53+p);idx=rng.permutation(n)
    ntrain=int(n*PARAMS['train_fraction']);train=idx[:ntrain];valid=idx[ntrain:]
    ks=kurtosis(Z[train],axis=0,fisher=True,bias=False)
    ks=np.nan_to_num(ks,nan=0.,posinf=99.,neginf=-99.)
    shortlist=np.argsort(-np.abs(ks),kind='stable')[:min(p,PARAMS['max_candidates'])]
    options=[]
    for j in shortlist:
        for m in (2,3):
            proposal=_test_gmm_1d(Z[train,j],Z[valid,j],m)
            if proposal is not None:
                options.append((proposal['bic_gain'],int(j),proposal))
    if not options:return None, {'reason':'no_heldout_mode'}
    score,j,choice=max(options,key=lambda x:x[0])
    out=np.searchsorted(choice['boundaries'],Z[:,j],side='right').astype(int)
    return out,{'reason':'rescued','coordinate':j, **choice}

def predict_cera(X,return_info=False):
    Z=canonicalize(X);previous=predict_scale_screen(X)
    kold=kcount(previous)
    # Preserve existing substantial multi-cluster assignments; rare rescue runs only
    if kold>2:return (previous,{'branch':'base_preserved','K':kold}) if return_info else previous
    rare,info=rescue_candidate(Z)
    if rare is not None and (kold<=1 or (kold==2 and info['K']>=3)):
        return (rare,{'branch':'validated_projection',**info}) if return_info else rare
    return (previous,{'branch':'base_partition','K':kold,'rescue':info}) if return_info else previous

