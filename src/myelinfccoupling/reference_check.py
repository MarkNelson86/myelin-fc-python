"""Compare a MATLAB_HELPER_CHECK.m output with the translated Python helpers."""
import json
from pathlib import Path
import numpy as np
from scipy.io import loadmat
from .preprocessing import mylog,nzmean,nzzscore,rmzeros,groupavg


def expected():
    a=np.array([0,.1,1,10,np.inf,np.nan])
    z,mu,sd=nzzscore(np.array([0,2,4,np.nan,np.inf]))
    rows,mask=rmzeros([np.array([[1,0],[3,4]]),np.array([[2,4],[np.nan,5]])])
    s=np.array([0,1,2,3,0,2,3,4,0,3,4,5,0,4,5,1000000]).reshape((2,2,4),order='F')
    return dict(log_unshifted=mylog(a),log_shifted=mylog(a,True),
                nzmean=nzmean(np.array([[0,2,4],[0,0,0],[np.nan,3,np.inf]]),axis=1),
                z=z,mu=mu,sd=sd,rows=rows,mask=mask,
                group_raw=groupavg(s,log_transform=False)[0],group_auto=groupavg(s)[0])


def compare(path,out):
    ref=loadmat(path,simplify_cells=True)['ref'];results={}
    for key,value in expected().items():
        actual=np.asarray(ref[key]);value=np.asarray(value)
        ok=actual.size==value.size and np.allclose(actual.reshape(-1),value.reshape(-1),rtol=1e-10,atol=1e-10,equal_nan=True)
        results[key]=dict(passed=bool(ok),matlab=actual.tolist(),python=value.tolist())
    report=dict(matlab_version=str(ref.get('matlab_version','unknown')),results=results,passed=all(r['passed'] for r in results.values()))
    Path(out).parent.mkdir(parents=True,exist_ok=True)
    Path(out).write_text(json.dumps(report,indent=2))
    return report


def main():
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mat',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();r=compare(a.mat,a.out)
    for key,entry in r['results'].items():print(f"{key}: {'PASS' if entry['passed'] else 'DIFFERS'}")
    print('Helper comparison only; full Figure 6 equivalence is a separate check.')
    raise SystemExit(0 if r['passed'] else 1)

if __name__=='__main__':main()
