import numpy as np
import pandas as pd
from scipy.io import loadmat,savemat
from myelinfccoupling.neural import SmallMLP,run
from myelinfccoupling.demo import make_subject_data


def test_small_mlp_learns_signal_and_is_deterministic():
    rng=np.random.default_rng(42);x=np.column_stack([rng.normal(size=(700,3)),rng.integers(0,2,700)])
    y=.2*x[:,0]-.1*x[:,1];weights=np.ones(700)
    model=SmallMLP(60,42,2).fit(x,y,weights)
    again=SmallMLP(60,42,2).fit(x,y,weights)
    np.testing.assert_allclose(model.predict(x),again.predict(x))
    assert np.mean((model.predict(x)-y)**2)<.2*np.var(y)
    np.testing.assert_allclose(model.scaler.mean_,x[:,:3].mean(0))


def test_neural_outer_targets_cannot_change_predictions_or_epochs(tmp_path):
    manifest,lut=make_subject_data(tmp_path/'inputs')
    out=tmp_path/'first';run(manifest,lut,out,n_outer=2,n_inner=2,candidates=[2],cap=80)
    state=np.load(out/'neural_fold_0.npz')
    assert state['hidden_weights'].shape[1]==16
    assert np.isfinite(state['input_scale']).all()
    folds=pd.read_csv(out/'outer_folds.csv');indices=folds[(folds.outer_fold==0)&(folds.role=='test')].scan_index.to_numpy()
    path=manifest.parent/'main/FC.mat';fc=loadmat(path)['Dts'];fc[:,:,indices]*=-1000;savemat(path,{'Dts':fc})
    changed=tmp_path/'changed';run(manifest,lut,changed,n_outer=2,n_inner=2,candidates=[2],cap=80)
    for filename,columns in [('predictions.csv',['NeuralNetwork','template']),('selected_epochs.csv',['epochs'])]:
        a=pd.read_csv(out/filename);b=pd.read_csv(changed/filename)
        np.testing.assert_allclose(a[a.outer_fold==0][columns],b[b.outer_fold==0][columns])
