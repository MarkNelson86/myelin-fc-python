"""Run a reproducible subject-modeling demo using entirely synthetic data."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.io import savemat
from .subjects import load_subject_manifest, audit_subjects, training_indices


def _save_mat(path, data):
    savemat(path, data)
    # MATLAB headers otherwise contain wall-clock time. Fix only descriptive
    # header bytes so regenerated synthetic input files have stable hashes.
    with path.open('r+b') as stream:
        stream.write(b'MATLAB 5.0 MAT-file: deterministic synthetic demo'.ljust(116,b' '))


def make_subject_data(out, seed=42):
    """Generate ten people, two held-out people with paired sessions, 24 nodes.

    Main and holdout contain matching first scans to exercise participant
    exclusion. These synthetic connectomes are workflow fixtures, not a model
    of biological measurements or evidence of scientific performance.
    """
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(seed);n=24;i,j=np.triu_indices(n,1);e=len(i)
    base=rng.lognormal(0,.4,(e,3));common=rng.normal(0,.15,e)
    scans={}
    for person in range(1,11):
        individual=rng.normal(0,.15,(e,3))
        for session in range(2 if person<=2 else 1):
            x=base*np.exp(individual+rng.normal(0,.025,(e,3)))
            present=rng.random(e)>.15
            fc=.3*x[:,0]-.2*x[:,1]+.1*x[:,2]+.1*x[:,0]*x[:,1]+common+rng.normal(0,.08,e)
            x[~present]=0
            values={}
            for key,vector in zip(['caliber','myelin','length','FC'],[*x.T,fc]):
                matrix=np.zeros((n,n));matrix[i,j]=vector;matrix[j,i]=vector
                values[key]=matrix
            scans[f'sub-{person:02d}'+('r' if session else '')]=values
    spec={'synthetic':True,'seed':seed,'batches':{}}
    for batch,ids in [('main',[f'sub-{p:02d}' for p in range(1,11)]),
                      ('holdout',['sub-01','sub-01r','sub-02','sub-02r'])]:
        folder=out/batch;folder.mkdir(exist_ok=True)
        _save_mat(folder/'ids.mat',{'Ss':np.array(ids,dtype=object)[None,:]})
        entries={'ids':{'path':f'{batch}/ids.mat','key':'Ss'}}
        for feature in ['caliber','myelin','length','FC']:
            _save_mat(folder/f'{feature}.mat',{'Dts':np.stack([scans[s][feature] for s in ids],axis=2)})
            entries[feature]={'path':f'{batch}/{feature}.mat','key':'Dts'}
        spec['batches'][batch]=entries
    manifest=out/'manifest.json';manifest.write_text(json.dumps(spec,indent=2)+'\n')
    labels=[f'7Networks_LH_{"Vis" if k<12 else "Default"}_{k+1}' for k in range(n)]
    lut=out/'networks.csv';pd.DataFrame({'label':labels}).to_csv(lut,index=False)
    return manifest,lut


def run_demo(out,seed=42,all_models=False):
    """Generate inputs, audit identities, run nested subject OLS/Ridge, summarize."""
    from .ridge import run
    out=Path(out)
    if out.exists() and any(out.iterdir()):
        raise ValueError(f'Output directory is not empty: {out}. Choose a new --out directory.')
    manifest,lut=make_subject_data(out/'inputs',seed)
    batches=load_subject_manifest(manifest)
    audit_subjects(batches).to_csv(out/'subject_audit.csv',index=False)
    exclusions=[]
    for person in dict.fromkeys(batches['holdout']['participant_ids']):
        keep=training_indices(batches['main'],person)
        exclusions.append(dict(participant_id=person,n_training=len(keep),
            excluded_scans=';'.join(s for s,p in zip(batches['main']['scan_ids'],batches['main']['participant_ids']) if p==person)))
    pd.DataFrame(exclusions).to_csv(out/'fold_exclusions.csv',index=False)
    metrics=run(manifest,lut,out/'models',alphas=[.01,1,100],n_inner=3,seed=seed,training_mode='subject')
    metrics=metrics[metrics.network_pair=='all']
    summary=metrics.groupby(['participant_id','model'])[['rmse','r','r_individual_deviation']].mean().groupby('model').mean()
    summary.to_csv(out/'summary.csv')
    (out/'DEMO_REPORT.md').write_text('# Synthetic subject-modeling demo\n\n'
        'All inputs are generated fixtures. Metrics demonstrate execution only; '
        'they are not scientific results. Ten synthetic people; two held-out people '
        'with paired sessions; nine training people per outer fold; three inner '
        'participant folds. Sessions are averaged within people before summary.\n\n'
        'See summary.csv, subject_audit.csv, fold_exclusions.csv and models/ for '
        'metrics, predictions, coefficients, scaling, selected alphas, fold '
        'assignments, input hashes and run configuration.\n')
    if all_models:
        from .cohort import run as cohort_run
        from .boosting import run as boosting_run
        from .neural import run as neural_run
        from .compare_runs import compare
        cohort_run(manifest,lut,out/'cohort-linear',n_outer=3,n_inner=2,seed=seed,alphas=[.01,1,100])
        boosting_run(manifest,lut,out/'cohort-boosting',n_outer=3,n_inner=2,seed=seed,candidates=[0,10],cap=200)
        neural_run(manifest,lut,out/'cohort-neural',n_outer=3,n_inner=2,seed=seed,candidates=[0,5],cap=200)
        compare({name:out/f'cohort-{name}'/'predictions.csv' for name in ['linear','boosting','neural']},out/'benchmark')
        with (out/'DEMO_REPORT.md').open('a') as stream:
            stream.write('\nThe optional all-model benchmark uses the ten main-stack people in three outer and two inner participant folds. Its separate benchmark/summary.csv compares residual OLS, Ridge, boosting and a small neural network on common edges. Budgets are smaller than the real-data protocol. Synthetic metrics demonstrate execution only.\n')
    print(summary.to_string())
    print(f'Synthetic demo complete: {out.resolve()}')
    return summary


def main():
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',default='out/demo');p.add_argument('--seed',type=int,default=42)
    p.add_argument('--all-models',action='store_true',help='Also compare residual linear, tree and neural cohort models')
    args=p.parse_args()
    try: run_demo(args.out,args.seed,args.all_models)
    except (ValueError,FileNotFoundError) as exc: p.error(str(exc))
    except ModuleNotFoundError as exc:
        if exc.name=='sklearn':p.error('Install modeling dependencies with: python -m pip install -e ".[ml]"')
        raise

if __name__=='__main__':main()
