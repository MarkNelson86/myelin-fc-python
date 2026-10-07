import hashlib
import pandas as pd
import pytest
import numpy as np
from myelinfccoupling.demo import make_subject_data,run_demo
from myelinfccoupling.subjects import load_subject_manifest


def test_synthetic_inputs_are_deterministic_and_include_rescans(tmp_path):
    first,_=make_subject_data(tmp_path/'a',seed=7)
    second,_=make_subject_data(tmp_path/'b',seed=7)
    for file in first.parent.rglob('*.mat'):
        counterpart=second.parent/file.relative_to(first.parent)
        assert hashlib.sha256(file.read_bytes()).digest()==hashlib.sha256(counterpart.read_bytes()).digest()
    b=load_subject_manifest(first)
    assert b['holdout']['participant_ids']==['sub-01','sub-01','sub-02','sub-02']
    for feature in b['main']['arrays']:
        np.testing.assert_array_equal(b['main']['arrays'][feature][:,:,0],b['holdout']['arrays'][feature][:,:,0])


def test_demo_runs_pipeline_and_protects_existing_outputs(tmp_path):
    out=tmp_path/'demo';summary=run_demo(out,seed=42,all_models=True)
    assert set(summary.index)=={'OLS','Ridge','Template'}
    assert np.isfinite(summary[['rmse','r']]).all().all()
    folds=pd.read_csv(out/'models/fold_assignments.csv')
    for (person,fold),d in folds.groupby(['outer_participant','inner_fold']):
        tr=set(d[d.role=='train'].participant_id);val=set(d[d.role=='validation'].participant_id)
        assert not tr&val and person not in tr|val
    chosen=pd.read_csv(out/'models/selected_alpha.csv')
    assert (chosen.n_training==9).all()
    assert (out/'models/input_sha256.json').exists()
    benchmark=pd.read_csv(out/'benchmark/summary.csv')
    assert set(benchmark.model)=={'Template','linear / OLS','linear / Ridge','boosting / Boosting','neural / NeuralNetwork'}
    assert (pd.read_csv(out/'benchmark/edge_coverage.csv').retained_fraction==1).all()
    with pytest.raises(ValueError,match='not empty'):run_demo(out)
