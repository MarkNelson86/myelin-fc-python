import numpy as np
from sklearn.linear_model import Ridge
from myelinfccoupling.ridge import Block, fit_subjects, choose_alpha


def test_weighted_block_matches_sklearn():
    rng=np.random.default_rng(9)
    x=rng.uniform(.1,4,(120,3));y=x[:,0]*x[:,1]+rng.normal(size=120)
    w=rng.uniform(.1,2,120);w=w*len(w)/w.sum()
    m=Block(x,y,w)
    ref=Ridge(alpha=11).fit(m.transform(x),y,sample_weight=w)
    np.testing.assert_allclose(m.predict(x,11),ref.predict(m.transform(x)),atol=1e-10)


def test_equal_participant_weight_with_unequal_edges_and_sessions():
    rng=np.random.default_rng(10)
    x=rng.uniform(.1,4,(100,3,2));y=rng.normal(size=(100,2))
    x[40:,:,0]=0
    pairs=np.array(['pair']*100);people=np.array(['a','b'])
    models,_,_=fit_subjects(x,y,np.arange(2),pairs,people)
    m=models['pair']
    np.testing.assert_allclose(m.ym,(y[:40,0].mean()+y[:,1].mean())/2)
    xx=np.concatenate([x,x[:,:,:1]],axis=2);yy=np.column_stack([y,y[:,0]])
    other,_,_=fit_subjects(xx,yy,np.arange(3),pairs,np.array(['a','b','a']))
    np.testing.assert_allclose(m.predict(x[:,:,1],0),other['pair'].predict(x[:,:,1],0),atol=1e-10)


def test_subject_nested_folds_exclude_people_and_use_training_template():
    rng=np.random.default_rng(11)
    x=rng.uniform(.1,4,(60,3,7));y=x[:,0,:]+rng.normal(size=(60,7))
    people=np.array(['a','b','b','c','d','e','heldout']);keep=np.arange(6)
    pairs=np.array(['pair']*60)
    alpha,scores,assign,_=choose_alpha(x,y,people,keep,pairs,[.1,10],3,8,'subject')
    model,mask,template=fit_subjects(x,y,keep,pairs,people)
    np.testing.assert_allclose(template,y[:,keep].mean(1))
    for f in range(3):
        tr={r['participant_id'] for r in assign if r['inner_fold']==f and r['role']=='train'}
        va={r['participant_id'] for r in assign if r['inner_fold']==f and r['role']=='validation'}
        assert not tr&va and 'heldout' not in tr|va
    prediction=model['pair'].predict(x[:,:,0],alpha)
    x[:,:,6]*=1e8;y[:,6]*=-1e8
    alpha2,scores2,_,_=choose_alpha(x,y,people,keep,pairs,[.1,10],3,8,'subject')
    model2,mask2,template2=fit_subjects(x,y,keep,pairs,people)
    assert alpha==alpha2
    np.testing.assert_allclose(scores.mse,scores2.mse)
    np.testing.assert_allclose(prediction,model2['pair'].predict(x[:,:,0],alpha2))
    np.testing.assert_array_equal(mask,mask2)
    np.testing.assert_allclose(template,template2)
