"""Group-trained, participant-held-out network-pair linear prediction baseline.

Uses raw nonzero structural averages and training-fitted scaling. This is a
corrected predictive baseline, not numerical reproduction of MATLAB Figure 6.
"""
from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd
from .subjects import load_subject_manifest, training_indices, audit_subjects


def correlation(a, b):
    valid = np.isfinite(a) & np.isfinite(b)
    a, b = np.asarray(a)[valid], np.asarray(b)[valid]
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def nonzero_mean(a):
    valid = np.isfinite(a) & (a != 0)
    count = valid.sum(axis=-1)
    return np.divide(np.where(valid, a, 0).sum(axis=-1), count,
                     out=np.full(count.shape, np.nan), where=count > 0)


def design(x):
    return np.column_stack([np.ones(len(x)), x, x[:, 0]*x[:, 1], x[:, 1]*x[:, 2]])


def fit_block(x, y, min_dfe=10):
    mu, sd = x.mean(axis=0), x.std(axis=0, ddof=1)
    if np.any(sd == 0):
        return None
    d = design((x-mu)/sd)
    rank = np.linalg.matrix_rank(d)
    if rank != d.shape[1] or len(y)-rank < min_dfe:
        return None
    beta = np.linalg.lstsq(d, y, rcond=None)[0]
    return mu, sd, beta


def read_networks(path, n_nodes):
    lut = pd.read_csv(path)
    lut = lut[lut['label'].str.startswith('7Networks_', na=False)]
    if len(lut) != n_nodes:
        raise ValueError(f'Expected {n_nodes} cortical labels, found {len(lut)}')
    names = lut['label'].str.split('_').str[2].to_numpy()
    allowed = ['Vis', 'SomMot', 'DorsAttn', 'SalVentAttn', 'Limbic', 'Cont', 'Default']
    if not set(names).issubset(allowed):
        raise ValueError('Unknown network label')
    return names


def run(manifest, lut, out, permutations=1000, seed=20261006):
    if permutations < 0:
        raise ValueError('permutations must be nonnegative')
    batches = load_subject_manifest(manifest)
    main, hold = batches['main'], batches['holdout']
    n = main['arrays']['FC'].shape[0]
    networks = read_networks(lut, n)
    i, j = np.triu_indices(n, 1)
    pairs = np.array(['|'.join(sorted((a,b))) for a,b in zip(networks[i], networks[j])])
    features = ['caliber', 'myelin', 'length']
    mx = np.stack([main['arrays'][f][i,j,:] for f in features], axis=1)
    hx = np.stack([hold['arrays'][f][i,j,:] for f in features], axis=1)
    my = main['arrays']['FC'][i,j,:]
    hy = hold['arrays']['FC'][i,j,:]
    rng = np.random.default_rng(seed)
    metrics, folds, coefficients, statuses, predictions, scaling = [], [], [], [], [], []
    for person in dict.fromkeys(hold['participant_ids']):
        keep = training_indices(main, person)
        if len(keep) < 2:
            raise ValueError('At least two training participants required')
        xg = nonzero_mean(mx[:,:,keep])
        # FC zero is a valid correlation; do not remove it as missing.
        yg = my[:,keep].mean(axis=1)
        trainmask = np.isfinite(xg).all(axis=1) & np.isfinite(yg)
        excluded = [s for s,p in zip(main['scan_ids'], main['participant_ids']) if p == person]
        folds.append(dict(participant_id=person, n_training=len(keep), excluded_scans=';'.join(excluded)))
        models = {}
        for pair in np.unique(pairs):
            mask = trainmask & (pairs == pair)
            models[pair] = fit_block(xg[mask], yg[mask])
            statuses.append(dict(participant_id=person, network_pair=pair, n_training_edges=int(mask.sum()), status='ok' if models[pair] else 'insufficient_or_rank_deficient'))
            if models[pair]:
                for idx, feature in enumerate(features):
                    scaling.append(dict(participant_id=person, network_pair=pair, feature=feature, mean=float(models[pair][0][idx]), std=float(models[pair][1][idx])))
                for term, value in zip(['intercept','caliber','myelin','length','caliber:myelin','myelin:length'],models[pair][2]):
                    coefficients.append(dict(participant_id=person, network_pair=pair, term=term, coefficient=float(value)))
        for scanidx in [k for k,p in enumerate(hold['participant_ids']) if p == person]:
            x, y = hx[:,:,scanidx], hy[:,scanidx]
            valid = trainmask & np.isfinite(x).all(axis=1) & (x != 0).all(axis=1) & np.isfinite(y)
            pred = np.full(len(y), np.nan)
            for pair, model in models.items():
                if model is None: continue
                mask = valid & (pairs == pair)
                mu, sd, beta = model
                pred[mask] = design((x[mask]-mu)/sd) @ beta
            valid &= np.isfinite(pred)
            permuted = np.full((permutations, len(y)), np.nan)
            # Shuffling stays within each network pair, including for pooled scores.
            for pair in np.unique(pairs):
                ids = np.flatnonzero(valid & (pairs == pair))
                for k in range(permutations):
                    permuted[k,ids] = rng.permutation(y[ids])
            for label in ['all', *np.unique(pairs)]:
                mask = valid if label == 'all' else valid & (pairs == label)
                yp, yt, template = pred[mask], y[mask], yg[mask]
                r = correlation(yp,yt)
                null = np.array([correlation(yp, p[mask]) for p in permuted])
                finite_null = null[np.isfinite(null)]
                other = np.array([correlation(yp,my[mask,k]) for k in keep])
                other = other[np.isfinite(other)]
                pperm = (1+np.sum(finite_null>=r))/(1+len(finite_null)) if np.isfinite(r) and len(finite_null) else np.nan
                ss = np.sum((yt-yt.mean())**2) if len(yt) else 0
                metrics.append(dict(participant_id=person, scan_id=hold['scan_ids'][scanidx], network_pair=label,
                    n_edges=int(mask.sum()), r_true=r, r_template=correlation(template,yt),
                    r_individual_deviation=correlation(yp-template,yt-template),
                    rmse=float(np.sqrt(np.mean((yp-yt)**2))) if len(yt) else np.nan,
                    rmse_template=float(np.sqrt(np.mean((template-yt)**2))) if len(yt) else np.nan,
                    R2=1-np.sum((yp-yt)**2)/ss if ss else np.nan,
                    p_permutation=pperm, r_other_mean=float(other.mean()) if len(other) else np.nan,
                    own_minus_other=r-float(other.mean()) if len(other) else np.nan,
                    own_percentile=float(np.mean(other<r)) if len(other) and np.isfinite(r) else np.nan))
            predictions.append(pd.DataFrame(dict(scan_id=hold['scan_ids'][scanidx], i=i[valid]+1,j=j[valid]+1,network_pair=pairs[valid],empirical=y[valid],predicted=pred[valid],training_mean=yg[valid])))
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    for filename, rows in [('metrics',metrics),('fold_exclusions',folds),('coefficients',coefficients),('model_status',statuses),('scaling',scaling)]:
        pd.DataFrame(rows).to_csv(out/f'{filename}.csv',index=False)
    pd.concat(predictions,ignore_index=True).to_csv(out/'predictions.csv',index=False)
    audit_subjects(batches).to_csv(out/'subject_audit.csv', index=False)
    spec = json.loads(Path(manifest).read_text())
    files = {Path(manifest).resolve(), Path(lut).resolve()}
    for batch in spec['batches'].values():
        files.update((Path(manifest).parent / batch[key]['path']).resolve() for key in ['ids', *features, 'FC'])
    provenance = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(files)}
    (out/'input_sha256.json').write_text(json.dumps(provenance,indent=2))
    (out/'run_config.json').write_text(json.dumps(dict(seed=seed,permutations=permutations,preprocessing='raw_nonzero_structural_mean; training_pair_scaling; raw_FC',mask='training_structural_union; strict_upper_triangle',myelin_manifest_key='myelin',specificity='descriptive comparison against training subjects'),indent=2))
    return pd.DataFrame(metrics)


def main():
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', required=True);p.add_argument('--lut',required=True)
    p.add_argument('--out',required=True);p.add_argument('--permutations',type=int,default=1000)
    p.add_argument('--seed',type=int,default=20261006)
    a=p.parse_args(); run(a.manifest,a.lut,a.out,a.permutations,a.seed)
    print(f'Results saved to {a.out}')

if __name__ == '__main__': main()
