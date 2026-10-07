"""Compare saved linear and boosting runs on common participant, scan and edge rows."""
from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd
from .generalizability import correlation

KEYS=['participant_id','scan_id','i','j']
MODEL_COLUMNS=['OLS','Ridge','Boosting','NeuralNetwork']


def compare(runs,out):
    """Require identical scan sets and matching empirical FC/templates.

    runs maps descriptive labels to predictions.csv paths. Compare the same
    cohort and exclusions; an intersected edge set is used for every model.
    """
    if len(runs)<2: raise ValueError('Provide at least two named runs')
    frames={};common=None;scan_sets=[];coverage=[]
    for label,path in runs.items():
        frame=pd.read_csv(path)
        columns=[c for c in MODEL_COLUMNS if c in frame]
        required=KEYS+['empirical','template']
        if not columns:raise ValueError(f'{label}: no recognized model prediction columns')
        if not set(required).issubset(frame.columns): raise ValueError(f'{label}: missing prediction columns')
        if frame[KEYS].isna().any().any() or frame.duplicated(KEYS).any(): raise ValueError(f'{label}: missing or duplicate edge keys')
        if not np.isfinite(frame[['empirical','template',*columns]].to_numpy()).all(): raise ValueError(f'{label}: nonfinite predictions')
        scan_sets.append(set(zip(frame.participant_id,frame.scan_id)))
        frame=frame.set_index(KEYS)
        frames[label]=frame
        common=frame.index if common is None else common.intersection(frame.index)
    if any(scans!=scan_sets[0] for scans in scan_sets): raise ValueError('Runs contain different participant/scan sets')
    if not len(common): raise ValueError('Runs have no common edges')
    aligned={label:frame.loc[common] for label,frame in frames.items()}
    reference=next(iter(aligned.values()))
    if set(zip(common.get_level_values(0),common.get_level_values(1)))!=scan_sets[0]:
        raise ValueError('At least one scan has no common edges')
    for label,frame in aligned.items():
        if not np.allclose(frame[['empirical','template']],reference[['empirical','template']],rtol=1e-10,atol=1e-10):
            raise ValueError(f'{label}: empirical FC or training template differs; compare matching data and exclusions')
        coverage.append(dict(run=label,input_edges=len(frames[label]),common_edges=len(common),retained_fraction=len(common)/len(frames[label])))
    records=[]
    for (person,scan),ref in reference.groupby(level=[0,1]):
        y=ref.empirical.to_numpy();template=ref.template.to_numpy()
        predictions={'Template':template}
        for label,frame in aligned.items():
            block=frame.loc[ref.index]
            for model in MODEL_COLUMNS:
                if model not in block:continue
                predictions[f'{label} / {model}']=block[model].to_numpy()
        for label,pred in predictions.items():
            denominator=np.sum((template-y)**2)
            records.append(dict(participant_id=person,scan_id=scan,model=label,n_edges=len(y),
                rmse=np.sqrt(np.mean((pred-y)**2)),r=correlation(pred,y),
                r_individual_deviation=correlation(pred-template,y-template),
                R2_vs_template=1-np.sum((pred-y)**2)/denominator if denominator>0 else np.nan))
    scores=pd.DataFrame(records)
    metrics=['rmse','r','r_individual_deviation','R2_vs_template']
    participants=scores.groupby(['participant_id','model'])[metrics].mean()
    summary=participants.groupby('model').mean()
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    scores.to_csv(out/'scan_metrics.csv',index=False)
    participants.to_csv(out/'participant_metrics.csv')
    summary.to_csv(out/'summary.csv');pd.DataFrame(coverage).to_csv(out/'edge_coverage.csv',index=False)
    config={'runs':{label:str(Path(path).resolve()) for label,path in runs.items()},
            'input_sha256':{label:hashlib.sha256(Path(path).read_bytes()).hexdigest() for label,path in runs.items()},
            'aggregation':'session metric mean within participant, then mean participants',
            'comparison':'intersection of participant/scan/edge keys; same empirical FC and template',
            'common_edge_rows':len(common)}
    (out/'comparison_config.json').write_text(json.dumps(config,indent=2)+'\n')
    print(summary.to_string());print(f'Compared {len(common)} common edge rows; results in {out}')
    return summary


def main():
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',action='append',required=True,metavar='LABEL=PATH')
    p.add_argument('--out',required=True);args=p.parse_args();runs={}
    for item in args.run:
        if '=' not in item:p.error('Each --run must be LABEL=PATH')
        label,path=item.split('=',1)
        if not label or not path or label in runs or label=='Template':p.error('Use unique nonempty labels other than Template')
        runs[label]=path
    try:compare(runs,args.out)
    except (ValueError,FileNotFoundError) as exc:p.error(str(exc))

if __name__=='__main__':main()
