"""Nested participant-fold comparison of group-trained OLS and Ridge."""
from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from .subjects import load_subject_manifest, training_indices
from .generalizability import nonzero_mean, correlation, design, read_networks

DEFAULT_ALPHAS=np.logspace(-4,6,11)


class Block:
    """Scale training main effects and all five terms; intercept is unpenalized.

    Cached sufficient statistics solve sklearn Ridge's sum-of-squares objective.
    No estimator or scaler is reused across participant folds.
    """
    def __init__(self,x,y):
        self.mu=x.mean(0);self.sd=x.std(0,ddof=1)
        if np.any(self.sd==0): raise ValueError('constant predictor')
        terms=design((x-self.mu)/self.sd)[:,1:]
        self.tm=terms.mean(0);self.ts=terms.std(0,ddof=1)
        if np.any(self.ts==0): raise ValueError('constant term')
        d=(terms-self.tm)/self.ts
        if np.linalg.matrix_rank(d)!=5 or len(y)-6<10:
            raise ValueError('insufficient or rank deficient')
        self.ym=y.mean(); self.gram=d.T@d;self.rhs=d.T@(y-self.ym)
        self.beta_ols=np.linalg.lstsq(d,y-self.ym,rcond=None)[0]

    def transform(self,x):
        return (design((x-self.mu)/self.sd)[:,1:]-self.tm)/self.ts

    def beta(self,alpha):
        if alpha==0: return self.beta_ols
        return np.linalg.solve(self.gram+alpha*np.eye(5),self.rhs)

    def predict(self,x,alpha): return self.ym+self.transform(x)@self.beta(alpha)


def fit_groups(x,y,keep,pairs):
    xg=nonzero_mean(x[:,:,keep]);yg=y[:,keep].mean(1)
    valid=np.isfinite(xg).all(1)&np.isfinite(yg)
    models={}
    for pair in np.unique(pairs):
        mask=valid&(pairs==pair)
        try: models[pair]=Block(xg[mask],yg[mask])
        except ValueError: models[pair]=None
    return models,valid,yg


def predict_all(models,trainmask,x,pairs,alpha):
    pred=np.full(len(x),np.nan)
    valid=trainmask&np.isfinite(x).all(1)&(x!=0).all(1)
    for pair,model in models.items():
        if model is not None:
            mask=valid&(pairs==pair)
            pred[mask]=model.predict(x[mask],alpha)
    return pred


def choose_alpha(x,y,people,keep,pairs,alphas,n_inner,seed):
    # Splitting unique identities permits multiple sessions without leakage.
    unique=np.array(sorted(set(people[k] for k in keep)))
    if len(unique)<n_inner: raise ValueError('Too few participants for inner folds')
    records=[];assignments=[]
    for fold,(itr,ival) in enumerate(KFold(n_inner,shuffle=True,random_state=seed).split(unique)):
        train=keep[np.isin(people[keep],unique[itr])]
        val=keep[np.isin(people[keep],unique[ival])]
        models,mask,_=fit_groups(x,y,train,pairs)
        for idx in train: assignments.append(dict(inner_fold=fold,scan_index=int(idx),participant_id=people[idx],role='train'))
        for idx in val: assignments.append(dict(inner_fold=fold,scan_index=int(idx),participant_id=people[idx],role='validation'))
        for alpha in alphas:
            for idx in val:
                pred=predict_all(models,mask,x[:,:,idx],pairs,alpha)
                valid=np.isfinite(pred)&np.isfinite(y[:,idx])
                if valid.sum()<3: raise ValueError('No eligible validation edges')
                records.append(dict(inner_fold=fold,participant_id=people[idx],alpha=float(alpha),
                                    mse=float(np.mean((pred[valid]-y[valid,idx])**2)),
                                    n_edges=int(valid.sum())))
    scores=pd.DataFrame(records)
    # First average sessions for each person, then average participant RMSE.
    subject=scores.groupby(['alpha','participant_id'],as_index=False).mse.mean()
    subject['rmse']=np.sqrt(subject.mse)
    mean=subject.groupby('alpha').rmse.mean()
    return float(mean.idxmin()),scores,assignments,mean.to_dict()


def run(manifest,lut,out,alphas=DEFAULT_ALPHAS,n_inner=5,seed=20261006):
    alphas=np.asarray(alphas,dtype=float)
    if alphas.ndim!=1 or not len(alphas) or not np.isfinite(alphas).all() or np.any(alphas<=0):
        raise ValueError('Provide a nonempty list of finite positive Ridge alphas')
    b=load_subject_manifest(manifest);main,hold=b['main'],b['holdout']
    n=main['arrays']['FC'].shape[0];i,j=np.triu_indices(n,1)
    networks=read_networks(lut,n)
    pairs=np.array(['|'.join(sorted((a,b))) for a,b in zip(networks[i],networks[j])])
    features=['caliber','myelin','length']
    mx=np.stack([main['arrays'][f][i,j,:] for f in features],axis=1)
    hx=np.stack([hold['arrays'][f][i,j,:] for f in features],axis=1)
    my=main['arrays']['FC'][i,j,:];hy=hold['arrays']['FC'][i,j,:]
    people=np.array(main['participant_ids'])
    rows=[];all_scores=[];folds=[];chosen=[];predictions=[];coeff=[];status=[];scalers=[]
    for person in dict.fromkeys(hold['participant_ids']):
        keep=training_indices(main,person)
        alpha,scores,assignments,means=choose_alpha(mx,my,people,keep,pairs,alphas,n_inner,seed)
        scores['outer_participant']=person;all_scores.append(scores)
        for rec in assignments: folds.append(dict(outer_participant=person,**rec))
        chosen.append(dict(participant_id=person,alpha=alpha,n_training=len(set(people[keep])),
                           excluded_scans=';'.join(s for s,p in zip(main['scan_ids'],people) if p==person)))
        models,mask,template=fit_groups(mx,my,keep,pairs)
        for pair,m in models.items():
            status.append(dict(participant_id=person,network_pair=pair,status='ok' if m else 'skipped'))
            if m:
                for k,f in enumerate(features):scalers.append(dict(participant_id=person,network_pair=pair,stage='main',feature=f,mean=m.mu[k],std=m.sd[k]))
                for k,f in enumerate(['caliber','myelin','length','caliber:myelin','myelin:length']):scalers.append(dict(participant_id=person,network_pair=pair,stage='design',feature=f,mean=m.tm[k],std=m.ts[k]))
                for label,a in [('OLS',0),('Ridge',alpha)]:
                    for term,value in zip(['caliber','myelin','length','caliber:myelin','myelin:length'],m.beta(a)):
                        coeff.append(dict(participant_id=person,network_pair=pair,model=label,term=term,coefficient=value,intercept=m.ym))
        for idx,p in enumerate(hold['participant_ids']):
            if p!=person:continue
            po=predict_all(models,mask,hx[:,:,idx],pairs,0)
            pr=predict_all(models,mask,hx[:,:,idx],pairs,alpha)
            valid=np.isfinite(po)&np.isfinite(pr)&np.isfinite(hy[:,idx])
            predictions.append(pd.DataFrame(dict(participant_id=person,scan_id=hold['scan_ids'][idx],i=i[valid]+1,j=j[valid]+1,network_pair=pairs[valid],empirical=hy[valid,idx],OLS=po[valid],Ridge=pr[valid],template=template[valid])))
            for group in ['all',*np.unique(pairs)]:
                use=valid if group=='all' else valid&(pairs==group)
                y=hy[use,idx];t=template[use]
                for label,pred in [('OLS',po[use]),('Ridge',pr[use]),('Template',t)]:
                    ss=np.sum((y-y.mean())**2) if len(y) else 0
                    rows.append(dict(participant_id=person,scan_id=hold['scan_ids'][idx],network_pair=group,model=label,alpha=alpha if label=='Ridge' else 0,n_edges=int(use.sum()),
                                     rmse=np.sqrt(np.mean((pred-y)**2)) if len(y) else np.nan,
                                     mae=np.mean(abs(pred-y)) if len(y) else np.nan,
                                     R2=1-np.sum((pred-y)**2)/ss if ss else np.nan,
                                     r=correlation(pred,y),r_individual_deviation=correlation(pred-t,y-t)))
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    for name,data in [('metrics',rows),('selected_alpha',chosen),('fold_assignments',folds),('coefficients',coeff),('model_status',status),('scaling',scalers)]:pd.DataFrame(data).to_csv(out/f'{name}.csv',index=False)
    pd.concat(all_scores,ignore_index=True).to_csv(out/'inner_scores.csv',index=False)
    pd.concat(predictions,ignore_index=True).to_csv(out/'predictions.csv',index=False)
    spec=json.loads(Path(manifest).read_text());files={Path(manifest).resolve(),Path(lut).resolve()}
    for batch in spec['batches'].values():files.update((Path(manifest).parent/batch[key]['path']).resolve() for key in ['ids',*features,'FC'])
    (out/'input_sha256.json').write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)},indent=2))
    (out/'run_config.json').write_text(json.dumps(dict(seed=seed,alphas=alphas.tolist(),inner_folds=n_inner,criterion='mean participant RMSE',training='raw nonzero structural group averages; raw FC',comparison='OLS and Ridge on identical eligible edges',sklearn_objective='sum squared residuals + alpha * squared slopes',historical_replication=False),indent=2))
    return pd.DataFrame(rows)


def main():
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',required=True);p.add_argument('--lut',required=True);p.add_argument('--out',required=True)
    p.add_argument('--inner-folds',type=int,default=5);p.add_argument('--seed',type=int,default=20261006)
    p.add_argument('--alphas',type=float,nargs='+',default=DEFAULT_ALPHAS.tolist())
    a=p.parse_args();run(a.manifest,a.lut,a.out,a.alphas,a.inner_folds,a.seed)
    print(f'Results saved to {a.out}')

if __name__=='__main__':main()
