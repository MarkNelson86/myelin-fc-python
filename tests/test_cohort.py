import json
import numpy as np
import pandas as pd
import pytest
from scipy.io import loadmat,savemat
from myelinfccoupling.demo import make_subject_data
from myelinfccoupling.cohort import outer_splits,run
from myelinfccoupling.compare_runs import compare


def test_outer_splits_group_sessions_and_test_every_scan_once():
    people=np.array(['a','a','b','c','d','d','e','f'])
    tested=[]
    for _,train,test in outer_splits(people,3,42):
        assert not set(people[train])&set(people[test])
        tested.extend(test)
    assert sorted(tested)==list(range(len(people)))


def test_single_cohort_nested_workflow_and_outer_target_invariance(tmp_path):
    manifest,lut=make_subject_data(tmp_path/'inputs')
    spec=json.loads(manifest.read_text());del spec['batches']['holdout']
    spec['batches']['main']['participant_ids']=[f'HC{k:03d}' for k in range(10)]
    manifest.write_text(json.dumps(spec))
    out=tmp_path/'residual';run(manifest,lut,out,n_outer=2,n_inner=2,alphas=[.1,10])
    direct=tmp_path/'direct';run(manifest,lut,direct,training_mode='subject',n_outer=2,n_inner=2,alphas=[.1,10])
    compare({'residual':out/'predictions.csv','direct':direct/'predictions.csv'},tmp_path/'compare')
    outer=pd.read_csv(out/'outer_folds.csv');inner=pd.read_csv(out/'inner_folds.csv')
    assert len(outer[outer.role=='test'])==10
    for fold,d in outer.groupby('outer_fold'):
        test=set(d[d.role=='test'].participant_id)
        ii=inner[inner.outer_fold==fold]
        assert not test&set(ii.participant_id)
        for _,f in ii.groupby('inner_fold'):
            assert not set(f[f.role=='train'].participant_id)&set(f[f.role=='validation'].participant_id)
    # Change outer-test FC only; this fold's predictions, references and tuning
    # must remain unchanged, although other folds legitimately change.
    test_indices=outer[(outer.outer_fold==0)&(outer.role=='test')].scan_index.to_numpy()
    fcpath=manifest.parent/'main/FC.mat';fc=loadmat(fcpath)['Dts']
    fc[:,:,test_indices]*=-1e6;savemat(fcpath,{'Dts':fc})
    changed=tmp_path/'changed';run(manifest,lut,changed,n_outer=2,n_inner=2,alphas=[.1,10])
    for filename,columns in [('predictions.csv',['OLS','Ridge','template']),('references.csv',['template','caliber_mean','myelin_mean','length_mean']),('selected_alpha.csv',['alpha'])]:
        a=pd.read_csv(out/filename);b=pd.read_csv(changed/filename)
        np.testing.assert_allclose(a[a.outer_fold==0][columns],b[b.outer_fold==0][columns],atol=1e-10)
    assert json.loads((out/'run_config.json').read_text())['external_transfer'] is False


def test_cohort_rejects_invalid_fold_counts_and_existing_outputs(tmp_path):
    manifest,lut=make_subject_data(tmp_path/'inputs')
    with pytest.raises(ValueError,match='Too few'):run(manifest,lut,tmp_path/'bad',n_outer=2,n_inner=9)
    assert not (tmp_path/'bad').exists()
    out=tmp_path/'used';out.mkdir();(out/'keep.txt').write_text('keep')
    with pytest.raises(ValueError,match='empty'):run(manifest,lut,out)
    assert (out/'keep.txt').read_text()=='keep'
