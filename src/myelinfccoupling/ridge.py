"""Nested participant folds for group, subject and template-deviation models."""
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

    Optional positive row weights also determine training scaling and centering.

    Cached sufficient statistics solve sklearn Ridge's sum-of-squares objective.
    No estimator or scaler is reused across participant folds.
    """
    def __init__(self,x,y,weights=None):
        if weights is None:
            self.mu=x.mean(0);self.sd=x.std(0,ddof=1)
        else:
            weights=np.asarray(weights,dtype=float)
            if weights.shape!=(len(y),) or not np.isfinite(weights).all() or np.any(weights<=0):
                raise ValueError('invalid weights')
            weights=weights*len(y)/weights.sum()
            self.mu=np.average(x,axis=0,weights=weights)
            self.sd=np.sqrt(np.average((x-self.mu)**2,axis=0,weights=weights))
        if np.any(self.sd==0): raise ValueError('constant predictor')
        terms=design((x-self.mu)/self.sd)[:,1:]
        if weights is None:
            self.tm=terms.mean(0);self.ts=terms.std(0,ddof=1)
        else:
            self.tm=np.average(terms,axis=0,weights=weights)
            self.ts=np.sqrt(np.average((terms-self.tm)**2,axis=0,weights=weights))
        if np.any(self.ts==0): raise ValueError('constant term')
        d=(terms-self.tm)/self.ts
        if np.linalg.matrix_rank(d)!=5 or len(y)-6<10:
            raise ValueError('insufficient or rank deficient')
        self.ym=y.mean() if weights is None else np.average(y,weights=weights)
        root=np.ones(len(y)) if weights is None else np.sqrt(weights)
        dw=d*root[:,None];yw=(y-self.ym)*root
        self.gram=dw.T@dw;self.rhs=dw.T@yw
        self.beta_ols=np.linalg.lstsq(dw,yw,rcond=None)[0]

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


class ResidualModels(dict):
    """Network models with edge-wise references fitted only to training scans."""
    def __init__(self,structural_reference,template):
        super().__init__()
        self.structural_reference=structural_reference
        self.template=template


def fit_subjects(x,y,keep,pairs,people,residual=False):
    """Pool participant-edge rows; equal total participant weight per block.

    Each person's sessions share their weight equally among sessions with valid
    rows in that block. Weights sum to row count, preserving Ridge's SSE scale.
    The training FC template is still the raw training-scan mean. With residual
    enabled, fit deviations from training edge means, preserving raw availability.
    """
    xg=nonzero_mean(x[:,:,keep]);template=y[:,keep].mean(1)
    trainmask=np.isfinite(xg).all(1)&np.isfinite(template)
    models=ResidualModels(xg,template) if residual else {}
    complete={idx: np.isfinite(x[:,:,idx]).all(1)&(x[:,:,idx]!=0).all(1)&np.isfinite(y[:,idx]) for idx in keep}
    for pair in np.unique(pairs):
        pairmask=trainmask&(pairs==pair)
        rows=[];targets=[];identities=[]
        for idx in keep:
            valid=pairmask&complete[idx]
            if valid.any():
                rows.append(x[valid,:,idx]-xg[valid] if residual else x[valid,:,idx])
                targets.append(y[valid,idx]-template[valid] if residual else y[valid,idx])
                identities.append(people[idx])
        if not rows:
            models[pair]=None
            continue
        counts={p:identities.count(p) for p in set(identities)}
        weights=np.concatenate([np.full(len(r),1/(counts[p]*len(r))) for r,p in zip(rows,identities)])
        try: models[pair]=Block(np.concatenate(rows),np.concatenate(targets),weights)
        except ValueError: models[pair]=None
    return models,trainmask,template


def fit_training(x,y,keep,pairs,people,training_mode):
    if training_mode=='group': return fit_groups(x,y,keep,pairs)
    if training_mode=='subject': return fit_subjects(x,y,keep,pairs,people)
    if training_mode=='residual': return fit_subjects(x,y,keep,pairs,people,residual=True)
    raise ValueError('training_mode must be group, subject or residual')


def predict_all(models,trainmask,x,pairs,alpha):
    pred=np.full(len(x),np.nan)
    valid=trainmask&np.isfinite(x).all(1)&(x!=0).all(1)
    for pair,model in models.items():
        if model is not None:
            mask=valid&(pairs==pair)
            if isinstance(models,ResidualModels):
                deviation=model.predict(x[mask]-models.structural_reference[mask],alpha)
                pred[mask]=models.template[mask]+deviation
            else:
                pred[mask]=model.predict(x[mask],alpha)
    return pred


def choose_alpha(x,y,people,keep,pairs,alphas,n_inner,seed,training_mode="group"):
    # Splitting unique identities permits multiple sessions without leakage.
    unique=np.array(sorted(set(people[k] for k in keep)))
    if len(unique)<n_inner: raise ValueError('Too few participants for inner folds')
    records=[];assignments=[]
    for fold,(itr,ival) in enumerate(KFold(n_inner,shuffle=True,random_state=seed).split(unique)):
        train=keep[np.isin(people[keep],unique[itr])]
        val=keep[np.isin(people[keep],unique[ival])]
        models,mask,_=fit_training(x,y,train,pairs,people,training_mode)
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


def run(manifest,lut,out,alphas=DEFAULT_ALPHAS,n_inner=5,seed=20261006,training_mode="group"):
    if training_mode not in {"group","subject","residual"}: raise ValueError("Unknown training mode")
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
    rows=[];all_scores=[];folds=[];chosen=[];predictions=[];coeff=[];status=[];scalers=[];references=[]
    for person in dict.fromkeys(hold['participant_ids']):
        print(f'Outer fold {person}: {training_mode} training and inner alpha selection',flush=True)
        keep=training_indices(main,person)
        alpha,scores,assignments,means=choose_alpha(mx,my,people,keep,pairs,alphas,n_inner,seed,training_mode)
        scores['outer_participant']=person;all_scores.append(scores)
        for rec in assignments: folds.append(dict(outer_participant=person,**rec))
        chosen.append(dict(participant_id=person,alpha=alpha,n_training=len(set(people[keep])),
                           excluded_scans=';'.join(s for s,p in zip(main['scan_ids'],people) if p==person)))
        models,mask,template=fit_training(mx,my,keep,pairs,people,training_mode)
        if isinstance(models,ResidualModels):
            references.append(pd.DataFrame(dict(participant_id=person,i=i[mask]+1,j=j[mask]+1,
                template=template[mask],caliber_mean=models.structural_reference[mask,0],
                myelin_mean=models.structural_reference[mask,1],length_mean=models.structural_reference[mask,2])))
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
            predictions.append(pd.DataFrame(dict(participant_id=person,scan_id=hold['scan_ids'][idx],i=i[valid]+1,j=j[valid]+1,network_pair=pairs[valid],empirical=hy[valid,idx],OLS=po[valid],Ridge=pr[valid],template=template[valid],OLS_deviation=po[valid]-template[valid],Ridge_deviation=pr[valid]-template[valid])))
            for group in ['all',*np.unique(pairs)]:
                use=valid if group=='all' else valid&(pairs==group)
                y=hy[use,idx];t=template[use]
                for label,pred in [('OLS',po[use]),('Ridge',pr[use]),('Template',t)]:
                    ss=np.sum((y-y.mean())**2) if len(y) else 0
                    rows.append(dict(participant_id=person,scan_id=hold['scan_ids'][idx],network_pair=group,model=label,alpha=alpha if label=='Ridge' else 0,n_edges=int(use.sum()),
                                     rmse=np.sqrt(np.mean((pred-y)**2)) if len(y) else np.nan,
                                     mae=np.mean(abs(pred-y)) if len(y) else np.nan,
                                     R2=1-np.sum((pred-y)**2)/ss if ss else np.nan,
                                     R2_vs_template=(1-np.sum((pred-y)**2)/np.sum((t-y)**2)) if len(y) and np.sum((t-y)**2)>0 else np.nan,
                                     r=correlation(pred,y),r_individual_deviation=correlation(pred-t,y-t)))
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    for name,data in [('metrics',rows),('selected_alpha',chosen),('fold_assignments',folds),('coefficients',coeff),('model_status',status),('scaling',scalers)]:pd.DataFrame(data).to_csv(out/f'{name}.csv',index=False)
    pd.concat(all_scores,ignore_index=True).to_csv(out/'inner_scores.csv',index=False)
    pd.concat(predictions,ignore_index=True).to_csv(out/'predictions.csv',index=False)
    if references: pd.concat(references,ignore_index=True).to_csv(out/'references.csv',index=False)
    spec=json.loads(Path(manifest).read_text());files={Path(manifest).resolve(),Path(lut).resolve()}
    for batch in spec['batches'].values():files.update((Path(manifest).parent/batch[key]['path']).resolve() for key in ['ids',*features,'FC'])
    (out/'input_sha256.json').write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)},indent=2))
    (out/'run_config.json').write_text(json.dumps(dict(seed=seed,alphas=alphas.tolist(),inner_folds=n_inner,criterion='mean participant RMSE',training=('raw nonzero structural group averages; raw FC' if training_mode=='group' else ('participant-edge rows; FC and SC deviations from training edge means; equal participant weight per network pair' if training_mode=='residual' else 'participant-edge rows; raw FC; equal participant weight per network pair')),training_mode=training_mode,weight_normalization=('none' if training_mode=='group' else 'sum weights equals number of training rows per block'),comparison='OLS and Ridge on identical eligible edges',sklearn_objective='sum squared residuals + alpha * squared slopes',historical_replication=False),indent=2))
    return pd.DataFrame(rows)


def main():
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',required=True);p.add_argument('--lut',required=True);p.add_argument('--out',required=True)
    p.add_argument('--inner-folds',type=int,default=5);p.add_argument('--seed',type=int,default=20261006)
    p.add_argument('--alphas',type=float,nargs='+',default=DEFAULT_ALPHAS.tolist())
    p.add_argument('--training-mode',choices=['group','subject','residual'],default='group')
    a=p.parse_args();run(a.manifest,a.lut,a.out,a.alphas,a.inner_folds,a.seed,a.training_mode)
    print(f'Results saved to {a.out}')

if __name__=='__main__':main()
