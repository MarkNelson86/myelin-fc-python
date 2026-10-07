"""Bounded gradient boosting of FC departures with nested participant folds."""
from dataclasses import dataclass
from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold
from threadpoolctl import threadpool_limits
from .subjects import load_subject_manifest,audit_subjects
from .generalizability import nonzero_mean,read_networks,correlation
from .cohort import outer_splits


@dataclass
class Fitted:
    estimator: object
    structural_reference: np.ndarray
    template: np.ndarray
    trainmask: np.ndarray
    pair_codes: np.ndarray

    def predict(self,x):
        pred=np.full(len(x),np.nan)
        valid=self.trainmask&np.isfinite(x).all(1)&(x!=0).all(1)
        if not valid.any():return pred
        features=np.column_stack([x[valid]-self.structural_reference[valid],self.pair_codes[valid]])
        with threadpool_limits(limits=1):
            correction=np.zeros(valid.sum()) if self.estimator is None else self.estimator.predict(features)
        pred[valid]=self.template[valid]+correction
        return pred


def fit(x,y,people,keep,pairs,iterations,cap,seed):
    """Training-only references, participant-balanced samples and weights.

    One shared tree ensemble conditions on categorical network pair. Each valid
    session gets an equal share of its person's total fitting weight. Sampling
    caps the total number of rows per person; evaluation uses full eligible rows.
    """
    reference=nonzero_mean(x[:,:,keep]);template=y[:,keep].mean(1)
    mask=np.isfinite(reference).all(1)&np.isfinite(template)
    codes=np.unique(pairs,return_inverse=True)[1]
    fitted=Fitted(None,reference,template,mask,codes)
    if iterations==0:return fitted,[]
    rng=np.random.default_rng(seed);features=[];targets=[];weights=[];records=[]
    for person in sorted(set(people[keep])):
        scans=keep[people[keep]==person];valid_scans=[]
        for idx in scans:
            valid=mask&np.isfinite(x[:,:,idx]).all(1)&(x[:,:,idx]!=0).all(1)&np.isfinite(y[:,idx])
            available=np.flatnonzero(valid)
            if len(available):valid_scans.append((idx,available))
        if not valid_scans:continue
        if cap<len(valid_scans):raise ValueError('Row cap must cover every valid session of a participant')
        per_scan=cap//len(valid_scans)
        for idx,available in valid_scans:
            selected=np.sort(rng.choice(available,size=min(len(available),per_scan),replace=False))
            features.append(np.column_stack([x[selected,:,idx]-reference[selected],codes[selected]]))
            targets.append(y[selected,idx]-template[selected])
            weights.append(np.full(len(selected),1/(len(valid_scans)*len(selected))))
            records.append(dict(participant_id=person,scan_index=int(idx),available_rows=len(available),sampled_rows=len(selected),edge_index_sha256=hashlib.sha256(selected.astype('<i8').tobytes()).hexdigest()))
    if not features:raise ValueError('No eligible training rows')
    xx=np.concatenate(features);yy=np.concatenate(targets);ww=np.concatenate(weights);ww*=len(ww)/ww.sum()
    model=HistGradientBoostingRegressor(max_iter=int(iterations),learning_rate=.05,max_depth=3,max_leaf_nodes=7,min_samples_leaf=50,l2_regularization=10,categorical_features=[3],early_stopping=False,random_state=seed)
    with threadpool_limits(limits=1):model.fit(xx,yy,sample_weight=ww)
    fitted.estimator=model
    return fitted,records


def select(x,y,people,keep,pairs,candidates,cap,n_inner,seed):
    unique=np.array(sorted(set(people[keep])))
    records=[];assignments=[];sampling=[]
    for fold,(tr,va) in enumerate(KFold(n_inner,shuffle=True,random_state=seed).split(unique)):
        train=keep[np.isin(people[keep],unique[tr])];val=keep[np.isin(people[keep],unique[va])]
        for role,indices in [('train',train),('validation',val)]:
            for idx in indices:assignments.append(dict(inner_fold=fold,scan_index=int(idx),participant_id=people[idx],role=role))
        for iterations in candidates:
            model,sampled=fit(x,y,people,train,pairs,iterations,cap,seed+fold)
            for row in sampled:sampling.append(dict(inner_fold=fold,iterations=iterations,**row))
            for idx in val:
                pred=model.predict(x[:,:,idx]);valid=np.isfinite(pred)&np.isfinite(y[:,idx])
                if valid.sum()<3:raise ValueError('No eligible validation edges')
                records.append(dict(inner_fold=fold,participant_id=people[idx],iterations=iterations,mse=float(np.mean((pred[valid]-y[valid,idx])**2)),n_edges=int(valid.sum())))
    scores=pd.DataFrame(records)
    participant=scores.groupby(['iterations','participant_id']).mse.mean().pow(.5)
    means=participant.groupby('iterations').mean()
    # Prefer less complexity on exact ties, including the zero-correction model.
    chosen=int(means.sort_index().idxmin())
    return chosen,scores,assignments,sampling


def run(manifest,lut,out,batch='main',n_outer=5,n_inner=5,seed=20261006,candidates=(0,30,60),cap=2500):
    candidates=list(candidates)
    if not candidates or any(not isinstance(v,(int,np.integer)) or v<0 for v in candidates) or len(set(candidates))!=len(candidates):raise ValueError('Iterations must be distinct nonnegative integers')
    if cap<1:raise ValueError('Row cap must be positive')
    out=Path(out)
    if out.exists() and any(out.iterdir()):raise ValueError('Output directory must be empty')
    batches=load_subject_manifest(manifest)
    if batch not in batches:raise ValueError(f'Missing batch {batch!r}')
    data=batches[batch];people=np.array(data['participant_ids']);n=data['arrays']['FC'].shape[0];i,j=np.triu_indices(n,1)
    networks=read_networks(lut,n);pairs=np.array(['|'.join(sorted((a,b))) for a,b in zip(networks[i],networks[j])])
    features=['caliber','myelin','length'];x=np.stack([data['arrays'][f][i,j,:] for f in features],axis=1);y=data['arrays']['FC'][i,j,:]
    splits=list(outer_splits(people,n_outer,seed))
    if n_inner<2 or any(len(set(people[tr]))<n_inner for _,tr,_ in splits):raise ValueError('Too few training participants for inner folds')
    metrics=[];predictions=[];outer=[];inner=[];chosen=[];scores=[];sampling=[];references=[]
    for fold,train,test in splits:
        print(f'Boosting outer fold {fold+1}/{n_outer}: grouped inner selection',flush=True)
        for role,indices in [('train',train),('test',test)]:
            for idx in indices:outer.append(dict(outer_fold=fold,scan_index=int(idx),participant_id=people[idx],scan_id=data['scan_ids'][idx],role=role))
        iterations,score,assign,sampled=select(x,y,people,train,pairs,candidates,cap,n_inner,seed)
        chosen.append(dict(outer_fold=fold,iterations=iterations,selection='template' if iterations==0 else 'boosting',n_training=len(set(people[train]))))
        print(f'  selected iterations: {iterations}',flush=True)
        score['outer_fold']=fold;scores.append(score)
        for row in assign:inner.append(dict(outer_fold=fold,**row))
        for row in sampled:sampling.append(dict(outer_fold=fold,stage='inner',**row))
        model,sampled=fit(x,y,people,train,pairs,iterations,cap,seed)
        for row in sampled:sampling.append(dict(outer_fold=fold,stage='outer',inner_fold=-1,iterations=iterations,**row))
        mask=model.trainmask
        references.append(pd.DataFrame(dict(outer_fold=fold,i=i[mask]+1,j=j[mask]+1,template=model.template[mask],caliber_mean=model.structural_reference[mask,0],myelin_mean=model.structural_reference[mask,1],length_mean=model.structural_reference[mask,2])))
        for idx in test:
            pred=model.predict(x[:,:,idx]);valid=np.isfinite(pred)&np.isfinite(y[:,idx])
            if valid.sum()<3:raise ValueError('No eligible test edges')
            predictions.append(pd.DataFrame(dict(outer_fold=fold,participant_id=people[idx],scan_id=data['scan_ids'][idx],i=i[valid]+1,j=j[valid]+1,network_pair=pairs[valid],empirical=y[valid,idx],Boosting=pred[valid],template=model.template[valid],Boosting_deviation=pred[valid]-model.template[valid])))
            for group in ['all',*np.unique(pairs)]:
                use=valid if group=='all' else valid&(pairs==group);target=y[use,idx];template=model.template[use]
                for label,p in [('Boosting',pred[use]),('Template',template)]:
                    error=np.sum((p-target)**2);base=np.sum((template-target)**2)
                    metrics.append(dict(outer_fold=fold,participant_id=people[idx],scan_id=data['scan_ids'][idx],network_pair=group,model=label,iterations=iterations,n_edges=int(use.sum()),rmse=np.sqrt(error/len(target)) if len(target) else np.nan,r=correlation(p,target),r_individual_deviation=correlation(p-template,target-template),R2_vs_template=1-error/base if base else np.nan))
    out.mkdir(parents=True,exist_ok=True)
    for name,rows in [('metrics',metrics),('outer_folds',outer),('inner_folds',inner),('selected_iterations',chosen),('sampling',sampling)]:pd.DataFrame(rows).to_csv(out/f'{name}.csv',index=False)
    pd.concat(scores,ignore_index=True).to_csv(out/'inner_scores.csv',index=False)
    pd.concat(predictions,ignore_index=True).to_csv(out/'predictions.csv',index=False)
    pd.concat(references,ignore_index=True).to_csv(out/'references.csv',index=False)
    audit_subjects({batch:data}).to_csv(out/'subject_audit.csv',index=False)
    table=pd.DataFrame(metrics);summary=table[table.network_pair=='all'].groupby(['participant_id','model'])[['rmse','r','r_individual_deviation','R2_vs_template']].mean().groupby('model').mean();summary.to_csv(out/'summary.csv')
    spec=json.loads(Path(manifest).read_text());entries=spec['batches'][batch];files={Path(manifest).resolve(),Path(lut).resolve()}
    files.update((Path(manifest).parent/entries[f]['path']).resolve() for f in ['ids',*features,'FC'])
    (out/'input_sha256.json').write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)},indent=2)+'\n')
    config=dict(protocol='nested participant KFold; residual target',estimator='HistGradientBoostingRegressor; shared model with categorical network pair',batch=batch,outer_folds=n_outer,inner_folds=n_inner,seed=seed,iteration_candidates=candidates,max_rows_per_person=cap,learning_rate=.05,max_depth=3,max_leaf_nodes=7,min_samples_leaf=50,l2_regularization=10,early_stopping=False,threads=1,criterion='mean participant RMSE; iteration 0 predicts template',features=features+['categorical network pair'],weighting='equal total participant weight; equal shares across valid sessions; weights sum to fitted row count',evaluation='full eligible test rows',n_participants=len(set(people)),n_scans=len(people),external_transfer=False)
    (out/'run_config.json').write_text(json.dumps(config,indent=2)+'\n')
    print(summary.to_string());return table


def main():
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',required=True);p.add_argument('--lut',required=True);p.add_argument('--out',required=True);p.add_argument('--batch',default='main')
    p.add_argument('--outer-folds',type=int,default=5);p.add_argument('--inner-folds',type=int,default=5);p.add_argument('--seed',type=int,default=20261006)
    p.add_argument('--iterations',type=int,nargs='+',default=[0,30,60]);p.add_argument('--max-rows-per-person',type=int,default=2500);a=p.parse_args()
    try:run(a.manifest,a.lut,a.out,a.batch,a.outer_folds,a.inner_folds,a.seed,a.iterations,a.max_rows_per_person)
    except (ValueError,FileNotFoundError) as exc:p.error(str(exc))

if __name__=='__main__':main()
