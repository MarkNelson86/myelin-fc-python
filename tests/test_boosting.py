import numpy as np
import pandas as pd
from scipy.io import loadmat,savemat
from myelinfccoupling.boosting import fit,run
from myelinfccoupling.demo import make_subject_data


def test_sampling_balances_people_and_is_reproducible():
    rng=np.random.default_rng(42)
    x=rng.uniform(1,2,(120,3,4));y=rng.normal(size=(120,4))
    people=np.array(['a','a','b','c']);keep=np.arange(4);pairs=np.array(['A']*120)
    first,records=fit(x,y,people,keep,pairs,2,80,42)
    second,again=fit(x,y,people,keep,pairs,2,80,42)
    assert records==again
    totals=pd.DataFrame(records).groupby('participant_id').sampled_rows.sum()
    assert totals.to_dict()=={'a':80,'b':80,'c':80}
    np.testing.assert_allclose(first.predict(x[:,:,0]),second.predict(x[:,:,0]))


def test_zero_correction_accepts_zero_deviations_and_masks_missing_structure():
    x=np.ones((5,3,3));y=np.arange(15).reshape(5,3)/10
    fitted,_=fit(x,y,np.array(['a','b','c']),np.arange(3),np.array(['A']*5),0,10,42)
    np.testing.assert_allclose(fitted.predict(x[:,:,0]),y.mean(1))
    missing=x[:,:,0].copy();missing[0,0]=0
    assert np.isnan(fitted.predict(missing)[0])


def test_nested_boosting_excludes_outer_targets_from_fit_and_selection(tmp_path):
    manifest,lut=make_subject_data(tmp_path/'inputs')
    out=tmp_path/'first';run(manifest,lut,out,n_outer=2,n_inner=2,candidates=[0,2],cap=80)
    folds=pd.read_csv(out/'outer_folds.csv');inner=pd.read_csv(out/'inner_folds.csv')
    test=folds[(folds.outer_fold==0)&(folds.role=='test')]
    assert not set(test.participant_id)&set(inner[inner.outer_fold==0].participant_id)
    for _,group in inner.groupby(['outer_fold','inner_fold']):
        assert not set(group[group.role=='train'].participant_id)&set(group[group.role=='validation'].participant_id)
    fcpath=manifest.parent/'main/FC.mat';fc=loadmat(fcpath)['Dts']
    fc[:,:,test.scan_index.to_numpy()]*=-1000;savemat(fcpath,{'Dts':fc})
    changed=tmp_path/'changed';run(manifest,lut,changed,n_outer=2,n_inner=2,candidates=[0,2],cap=80)
    for filename,columns in [('predictions.csv',['Boosting','template']),('selected_iterations.csv',['iterations'])]:
        a=pd.read_csv(out/filename);b=pd.read_csv(changed/filename)
        np.testing.assert_allclose(a[a.outer_fold==0][columns],b[b.outer_fold==0][columns])
