"""Explicit translations of supplied MATLAB helpers; not a Figure 6 replica.

The positive-real mylog path uses an omit-NaN minimum for shifting. MATLAB
runtime equivalence still needs a reference output. MATLAB dimensions map to
zero-based NumPy axes; nzzscore uses axis=None for pooled standardization.
"""
import numpy as np
from scipy.stats import skew


def mylog(a, shift=False):
    a=np.array(a,dtype=float,copy=True)
    a[~np.isfinite(a) | (a==0)]=np.nan
    if np.any(a<0):
        raise ValueError('Negative inputs require MATLAB complex-log semantics; unsupported')
    a[a==1]=.9999
    out=np.log10(a)
    finite=out[np.isfinite(out)]
    if shift and len(finite) and finite.min()<0:
        out+=1.001*abs(finite.min())
    return out


def nzmean(a, axis=-1):
    a=np.asarray(a,dtype=float)
    valid=np.isfinite(a)&(a!=0)
    count=valid.sum(axis=axis)
    return np.divide(np.where(valid,a,0).sum(axis=axis),count,
                     out=np.zeros(count.shape,dtype=float),where=count>0)


def groupavg(a, axis=-1, average='nz', log_transform=True):
    a=np.array(a,dtype=float,copy=True)
    values=a[a!=0]
    transformed=bool(len(values)>2 and abs(skew(values,bias=True))>1.5 and log_transform)
    if transformed:
        a=mylog(a,shift=True)
        a[np.isnan(a)]=0
    if average=='nz': return nzmean(a,axis),transformed
    if average=='all': return np.where(np.isfinite(a),a,0).mean(axis=axis),transformed
    raise ValueError("average must be 'nz' or 'all'")


def nzzscore(a, axis=None):
    a=np.array(a,dtype=float,copy=True)
    a[~np.isfinite(a)|(a==0)]=np.nan
    with np.errstate(invalid='ignore',divide='ignore'):
        mu=np.nanmean(a,axis=axis,keepdims=True)
        sd=np.nanstd(a,axis=axis,ddof=1,keepdims=True)
        # MATLAB pooled zscore returns zero for a constant finite sample.
        out=(a-mu)/np.where(sd==0,1,sd)
    return out,np.squeeze(mu),np.squeeze(sd)


def rmzeros(arrays):
    d=np.column_stack([np.asarray(a).ravel(order='F') for a in arrays])
    d=np.where(np.isnan(d),0,d)
    mask=(d!=0).all(axis=1)
    return d[mask],mask
