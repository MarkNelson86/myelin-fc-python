import numpy as np
from sklearn.linear_model import Ridge
from myelinfccoupling.ridge import Block,choose_alpha
from myelinfccoupling.preprocessing import mylog,nzmean,nzzscore,rmzeros


def test_ridge_matches_sklearn_and_shrinks():
    rng=np.random.default_rng(8);x=rng.uniform(1,4,(100,3));y=x[:,0]*x[:,1]+rng.normal(size=100)
    m=Block(x,y)
    expected=Ridge(alpha=12).fit(m.transform(x),y)
    np.testing.assert_allclose(m.predict(x,12),expected.predict(m.transform(x)),atol=1e-11)
    assert np.linalg.norm(m.beta(1e6))<np.linalg.norm(m.beta(0))
    np.testing.assert_allclose(m.predict(x,1e-8),m.predict(x,0),atol=1e-8)


def test_nested_exclusion_and_participant_folds():
    rng=np.random.default_rng(2)
    x=rng.uniform(1,4,(80,3,7));y=x[:,0,:]+x[:,1,:]
    people=np.array(['a','b','b','c','d','e','excluded']);keep=np.arange(6)
    pairs=np.array(['Vis|Vis']*80)
    a,s,assign,_=choose_alpha(x,y,people,keep,pairs,[.1,10],3,7)
    for fold in range(3):
        train={r['participant_id'] for r in assign if r['inner_fold']==fold and r['role']=='train'}
        val={r['participant_id'] for r in assign if r['inner_fold']==fold and r['role']=='validation'}
        assert not train&val and 'excluded' not in train|val
    x[:,:,6]*=1e12;y[:,6]*=-1e9
    a2,s2,_,_=choose_alpha(x,y,people,keep,pairs,[.1,10],3,7)
    assert a==a2
    np.testing.assert_allclose(s.mse,s2.mse)


def test_preprocessing_boundaries():
    a=np.array([0,.1,1,10,np.inf])
    out=mylog(a,True)
    assert np.isnan(out[[0,4]]).all()
    np.testing.assert_allclose(out[1:4],[.001,1.001+np.log10(.9999),2.001])
    np.testing.assert_allclose(nzmean(np.array([[0,2,4],[0,0,0]])),[3,0])
    z,mu,sd=nzzscore(np.array([0,2,4]))
    np.testing.assert_allclose(z[1:],[-1/np.sqrt(2),1/np.sqrt(2)])
    d,mask=rmzeros([np.array([[1,0],[3,4]]),np.array([[2,4],[np.nan,5]])])
    assert mask.tolist()==[True,False,False,True]
    np.testing.assert_allclose(d,[[1,2],[4,5]])
