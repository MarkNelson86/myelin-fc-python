import numpy as np
from myelinfccoupling.ridge import fit_training,predict_all,choose_alpha


def data():
    rng=np.random.default_rng(13);edges=60;subjects=8
    base=rng.uniform(2,4,(edges,3));x=base[:,:,None]+rng.normal(0,.2,(edges,3,subjects))
    y=rng.normal(size=(edges,1))+.6*x[:,0,:]-.2*x[:,1,:]+.3*x[:,2,:]
    return x,y,np.array([f'p{k}' for k in range(subjects)]),np.array(['pair']*edges)


def test_residual_recovers_known_individual_effect():
    x,y,people,pairs=data();keep=np.arange(7)
    models,mask,template=fit_training(x,y,keep,pairs,people,'residual')
    pred=predict_all(models,mask,x[:,:,7],pairs,0)
    np.testing.assert_allclose(pred,y[:,7],atol=1e-10)
    assert np.mean((pred-y[:,7])**2)<np.mean((template-y[:,7])**2)*.01


def test_zero_structural_deviations_are_valid():
    x,y,people,pairs=data()
    models,mask,template=fit_training(x,y,np.arange(7),pairs,people,'residual')
    raw=models.structural_reference.copy()
    pred=predict_all(models,mask,raw,pairs,1)
    assert np.isfinite(pred).all()
    expected=template+models['pair'].predict(np.zeros_like(raw),1)
    np.testing.assert_allclose(pred,expected)
    raw[0,0]=0
    assert np.isnan(predict_all(models,mask,raw,pairs,1)[0])


def test_residual_nested_exclusion_and_training_references():
    x,y,people,pairs=data();keep=np.arange(7)
    a,s,assign,_=choose_alpha(x,y,people,keep,pairs,[.1,10],3,7,'residual')
    m,mask,template=fit_training(x,y,keep,pairs,people,'residual')
    np.testing.assert_allclose(m.structural_reference,x[:,:,keep].mean(2))
    np.testing.assert_allclose(template,y[:,keep].mean(1))
    before=predict_all(m,mask,x[:,:,0],pairs,a)
    x[:,:,7]*=1e9;y[:,7]*=-1e9
    b,t,_,_=choose_alpha(x,y,people,keep,pairs,[.1,10],3,7,'residual')
    m2,mask2,template2=fit_training(x,y,keep,pairs,people,'residual')
    assert a==b
    np.testing.assert_allclose(s.mse,t.mse)
    np.testing.assert_allclose(before,predict_all(m2,mask2,x[:,:,0],pairs,b))
    np.testing.assert_allclose(template,template2)
    assert all(r['participant_id']!='p7' for r in assign)
