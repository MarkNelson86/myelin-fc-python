import numpy as np
from myelinfccoupling.generalizability import nonzero_mean, fit_block, design, correlation
from myelinfccoupling.subjects import training_indices


def test_nonzero_mean_and_valid_signed_fc():
    a=np.array([[0,2,4],[0,0,0],[np.nan,3,0]])
    np.testing.assert_allclose(nonzero_mean(a),[3,np.nan,3],equal_nan=True)
    assert correlation(np.array([-1.,0,1]),np.array([-1.,0,1])) == 1


def test_recover_interactions_and_training_scaling():
    rng=np.random.default_rng(42)
    x=rng.normal(size=(100,3))
    mu,sd=x.mean(0),x.std(0,ddof=1)
    beta=np.array([1,2,-1,3,.7,-.4])
    y=design((x-mu)/sd)@beta
    m,s,b=fit_block(x,y)
    np.testing.assert_allclose(b,beta,atol=1e-12)
    new=x[:8]+2
    np.testing.assert_allclose(design((new-m)/s)@b,design((new-mu)/sd)@beta,atol=1e-12)
    assert fit_block(np.ones((100,3)),y) is None


def test_excluded_participant_cannot_change_group_features():
    rng=np.random.default_rng(1)
    stack=rng.uniform(size=(20,3,4))
    batch={'participant_ids':['a','b','a','c']}
    keep=training_indices(batch,'a')
    original=nonzero_mean(stack[:,:,keep])
    stack[:,:,[0,2]]=1e12
    np.testing.assert_array_equal(nonzero_mean(stack[:,:,keep]),original)


def test_full_run_excludes_both_sessions_and_is_target_independent(tmp_path):
    import json
    import pandas as pd
    from scipy.io import savemat
    from myelinfccoupling.generalizability import run
    rng=np.random.default_rng(9); n=20
    i,j=np.triu_indices(n,1)
    entries={}
    for batch,ids in [('main',['sub-01','sub-02r','sub-03','sub-04']),('holdout',['sub-02','sub-02r'])]:
        folder=tmp_path/batch;folder.mkdir()
        savemat(folder/'ids.mat',{'Ss':np.array(ids,dtype=object)})
        entries[batch]={'ids':{'path':f'{batch}/ids.mat','key':'Ss'}}
        features=rng.uniform(1,5,(len(i),3,len(ids)))
        y=features[:,0,:]+features[:,1,:]*features[:,2,:]
        for k,name in enumerate(['caliber','myelin','length','FC']):
            a=np.zeros((n,n,len(ids)));v=features[:,k,:] if k<3 else y
            a[i,j,:]=v;a[j,i,:]=v
            savemat(folder/f'{name}.mat',{'Dts':a})
            entries[batch][name]={'path':f'{batch}/{name}.mat'}
    manifest=tmp_path/'manifest.json';manifest.write_text(json.dumps({'batches':entries}))
    lut=tmp_path/'lut.csv'
    pd.DataFrame({'label':[f'7Networks_LH_Vis_{k}' for k in range(n)]}).to_csv(lut,index=False)
    run(manifest,lut,tmp_path/'first',permutations=0)
    # Massive changes to the excluded participant and to test FC must not alter predictions.
    from scipy.io import loadmat
    for name in ['caliber','myelin','length','FC']:
        path=tmp_path/'main'/f'{name}.mat';a=loadmat(path)['Dts'];a[:,:,1]*=1e6;savemat(path,{'Dts':a})
    path=tmp_path/'holdout'/'FC.mat';a=loadmat(path)['Dts'];a*=3;savemat(path,{'Dts':a})
    run(manifest,lut,tmp_path/'second',permutations=0)
    first=pd.read_csv(tmp_path/'first'/'predictions.csv');second=pd.read_csv(tmp_path/'second'/'predictions.csv')
    np.testing.assert_allclose(first.predicted,second.predicted)
    assert set(first.scan_id)=={'sub-02','sub-02r'}
    exclusions=pd.read_csv(tmp_path/'first'/'fold_exclusions.csv')
    assert exclusions.excluded_scans.iloc[0]=='sub-02r'
