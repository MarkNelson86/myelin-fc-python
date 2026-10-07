"""Nested participant cross-validation for one cohort, including MICs-style stacks."""
from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from .subjects import load_subject_manifest,audit_subjects
from .generalizability import read_networks,correlation
from .ridge import DEFAULT_ALPHAS,choose_alpha,fit_training,predict_all,ResidualModels


def outer_splits(people,n_folds,seed):
    """Return scan indices while keeping every session of a person together."""
    people=np.asarray(people);unique=np.array(sorted(set(people)))
    if n_folds<2 or len(unique)<n_folds:raise ValueError('Outer folds must be between 2 and the participant count')
    for fold,(train,test) in enumerate(KFold(n_folds,shuffle=True,random_state=seed).split(unique)):
        yield fold,np.flatnonzero(np.isin(people,unique[train])),np.flatnonzero(np.isin(people,unique[test]))


def run(manifest,lut,out,batch='main',training_mode='residual',n_outer=5,n_inner=5,seed=20261006,alphas=DEFAULT_ALPHAS):
    if training_mode not in {'subject','residual'}:raise ValueError('Use subject or residual training mode')
    alphas=np.asarray(alphas,dtype=float)
    if not len(alphas) or alphas.ndim!=1 or not np.isfinite(alphas).all() or np.any(alphas<=0):raise ValueError('Alphas must be finite and positive')
    out=Path(out)
    if out.exists() and any(out.iterdir()):raise ValueError('Output directory must be empty; choose a new --out path')
    batches=load_subject_manifest(manifest)
    if batch not in batches:raise ValueError(f'Manifest has no batch {batch!r}')
    data=batches[batch];people=np.array(data['participant_ids'])
    n=data['arrays']['FC'].shape[0];i,j=np.triu_indices(n,1)
    networks=read_networks(lut,n)
    pairs=np.array(['|'.join(sorted((a,b))) for a,b in zip(networks[i],networks[j])])
    features=['caliber','myelin','length']
    x=np.stack([data['arrays'][f][i,j,:] for f in features],axis=1);y=data['arrays']['FC'][i,j,:]
    splits=list(outer_splits(people,n_outer,seed))
    if n_inner<2 or any(len(set(people[train]))<n_inner for _,train,_ in splits):raise ValueError('Too few outer-training participants for inner folds')
    metrics=[];predictions=[];outer=[];inner=[];chosen=[];scores=[];coefficients=[];scaling=[];status=[];references=[]
    for fold,train,test in splits:
        print(f'Outer fold {fold+1}/{n_outer}: {len(set(people[train]))} train, {len(set(people[test]))} test participants',flush=True)
        for role,indices in [('train',train),('test',test)]:
            for idx in indices:outer.append(dict(outer_fold=fold,scan_index=int(idx),scan_id=data['scan_ids'][idx],participant_id=people[idx],role=role))
        alpha,score,assign,_=choose_alpha(x,y,people,train,pairs,alphas,n_inner,seed,training_mode)
        chosen.append(dict(outer_fold=fold,alpha=alpha,n_training=len(set(people[train])),n_test=len(set(people[test]))))
        score['outer_fold']=fold;scores.append(score)
        for row in assign:inner.append(dict(outer_fold=fold,**row))
        models,mask,template=fit_training(x,y,train,pairs,people,training_mode)
        if isinstance(models,ResidualModels):
            references.append(pd.DataFrame(dict(outer_fold=fold,i=i[mask]+1,j=j[mask]+1,template=template[mask],caliber_mean=models.structural_reference[mask,0],myelin_mean=models.structural_reference[mask,1],length_mean=models.structural_reference[mask,2])))
        for pair,m in models.items():
            status.append(dict(outer_fold=fold,network_pair=pair,status='ok' if m else 'skipped'))
            if m is None:continue
            terms=['caliber','myelin','length','caliber:myelin','myelin:length']
            for k,feature in enumerate(features):scaling.append(dict(outer_fold=fold,network_pair=pair,stage='main',feature=feature,mean=m.mu[k],std=m.sd[k]))
            for k,term in enumerate(terms):scaling.append(dict(outer_fold=fold,network_pair=pair,stage='design',feature=term,mean=m.tm[k],std=m.ts[k]))
            for label,a in [('OLS',0),('Ridge',alpha)]:
                for term,beta in zip(terms,m.beta(a)):coefficients.append(dict(outer_fold=fold,network_pair=pair,model=label,term=term,coefficient=beta,intercept=m.ym))
        for idx in test:
            po=predict_all(models,mask,x[:,:,idx],pairs,0);pr=predict_all(models,mask,x[:,:,idx],pairs,alpha)
            valid=np.isfinite(po)&np.isfinite(pr)&np.isfinite(y[:,idx])
            if valid.sum()<3:raise ValueError(f'No eligible test edges for {data["scan_ids"][idx]}')
            predictions.append(pd.DataFrame(dict(outer_fold=fold,participant_id=people[idx],scan_id=data['scan_ids'][idx],i=i[valid]+1,j=j[valid]+1,network_pair=pairs[valid],empirical=y[valid,idx],OLS=po[valid],Ridge=pr[valid],template=template[valid],OLS_deviation=po[valid]-template[valid],Ridge_deviation=pr[valid]-template[valid])))
            for group in ['all',*np.unique(pairs)]:
                use=valid if group=='all' else valid&(pairs==group)
                target=y[use,idx];t=template[use];ss=np.sum((target-target.mean())**2) if len(target) else 0
                for label,pred in [('OLS',po[use]),('Ridge',pr[use]),('Template',t)]:
                    error=np.sum((pred-target)**2);baseline=np.sum((t-target)**2)
                    metrics.append(dict(outer_fold=fold,participant_id=people[idx],scan_id=data['scan_ids'][idx],network_pair=group,model=label,alpha=alpha if label=='Ridge' else 0,n_edges=int(use.sum()),rmse=np.sqrt(error/len(target)) if len(target) else np.nan,mae=np.mean(abs(pred-target)) if len(target) else np.nan,R2=1-error/ss if ss else np.nan,R2_vs_template=1-error/baseline if baseline else np.nan,r=correlation(pred,target),r_individual_deviation=correlation(pred-t,target-t)))
    out.mkdir(parents=True,exist_ok=True)
    table=pd.DataFrame(metrics)
    for name,rows in [('metrics',metrics),('outer_folds',outer),('inner_folds',inner),('selected_alpha',chosen),('coefficients',coefficients),('scaling',scaling),('model_status',status)]:pd.DataFrame(rows).to_csv(out/f'{name}.csv',index=False)
    pd.concat(predictions,ignore_index=True).to_csv(out/'predictions.csv',index=False)
    pd.concat(scores,ignore_index=True).to_csv(out/'inner_scores.csv',index=False)
    if references:pd.concat(references,ignore_index=True).to_csv(out/'references.csv',index=False)
    audit_subjects({batch:data}).to_csv(out/'subject_audit.csv',index=False)
    summary=table[table.network_pair=='all'].groupby(['participant_id','model'])[['rmse','r','r_individual_deviation','R2_vs_template']].mean().groupby('model').mean()
    summary.to_csv(out/'summary.csv')
    spec=json.loads(Path(manifest).read_text());entries=spec['batches'][batch]
    files={Path(manifest).resolve(),Path(lut).resolve()}
    files.update((Path(manifest).parent/entries[f]['path']).resolve() for f in ['ids',*features,'FC'])
    (out/'input_sha256.json').write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)},indent=2)+'\n')
    config=dict(protocol='nested participant KFold within one cohort',batch=batch,training_mode=training_mode,outer_folds=n_outer,inner_folds=n_inner,seed=seed,alphas=alphas.tolist(),criterion='mean participant RMSE',n_participants=len(set(people)),n_scans=len(people),aggregation='session metric mean within participant, then participant mean',historical_replication=False,external_transfer=False)
    (out/'run_config.json').write_text(json.dumps(config,indent=2)+'\n')
    print(summary.to_string());print(f'Cohort results saved to {out}')
    return table


def main():
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',required=True);p.add_argument('--lut',required=True);p.add_argument('--out',required=True)
    p.add_argument('--batch',default='main');p.add_argument('--training-mode',choices=['subject','residual'],default='residual')
    p.add_argument('--outer-folds',type=int,default=5);p.add_argument('--inner-folds',type=int,default=5);p.add_argument('--seed',type=int,default=20261006)
    p.add_argument('--alphas',type=float,nargs='+',default=DEFAULT_ALPHAS.tolist());a=p.parse_args()
    try:run(a.manifest,a.lut,a.out,a.batch,a.training_mode,a.outer_folds,a.inner_folds,a.seed,a.alphas)
    except (ValueError,FileNotFoundError) as exc:p.error(str(exc))

if __name__=='__main__':main()
