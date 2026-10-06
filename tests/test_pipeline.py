from pathlib import Path
import importlib.util
import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner
from myelinfccoupling import model, binning
from myelinfccoupling.config import Config
from myelinfccoupling.io import load_inputs
from myelinfccoupling.cli import main
from myelinfccoupling.util import sanitize_label

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('demo', ROOT / 'examples/make_demo_data.py')
demo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(demo)

@pytest.fixture
def edges():
    return demo.make_edges()

@pytest.mark.parametrize('interactions,filename', [(True,'global_full.csv'),(False,'global_reduced.csv')])
def test_supplied_python_global_reference(interactions, filename):
    # Independent saved outputs from the supplied original repository.
    actual = model.run(load_inputs(ROOT/'data/edges_fc_BOLDin.csv'), 'global', interactions)
    expected = pd.read_csv(ROOT/'tests/fixtures'/filename)
    for c in expected.select_dtypes(include='number'):
        np.testing.assert_allclose(actual[c], expected[c], atol=1e-10, rtol=1e-8)

@pytest.mark.parametrize('function,filename', [(binning.run_binned,'global_bins.csv'),(binning.bin_correlations,'global_bin_corr.csv')])
def test_supplied_python_binned_reference(function, filename):
    E = load_inputs(ROOT/'data/edges_fc_BOLDin.csv')
    actual = function(E, 'global') if function is binning.run_binned else function(E)
    expected = pd.read_csv(ROOT/'tests/fixtures'/filename)
    assert len(actual) == len(expected)
    for c in expected.select_dtypes(include='number'):
        np.testing.assert_allclose(actual[c], expected[c], atol=1e-10, rtol=1e-8)


def test_recovers_known_interaction_coefficients(edges):
    E = edges.copy()
    E.FC = 2 + 0.5*E.caliber - 0.3*E.myelin + 0.2*E.length + 0.4*E.myelin*E.caliber
    cfg = Config(standardize_predictors=False, standardize_response=False)
    r = model.run(E, 'global', cfg=cfg).iloc[0]
    np.testing.assert_allclose([r.B_Intercept,r.B_caliber,r.B_myelin,r.B_length,r.B_myelin_x_caliber,r.B_myelin_x_length], [2,.5,-.3,.2,.4,0], atol=1e-10)
    assert r.R2 == pytest.approx(1)


def test_group_counts_and_undirected_orientation(edges):
    node = model.run(edges, 'nodewise')
    assert len(node) == 24
    assert (node.n_obs == 23).all()
    assert len(model.run(edges, 'rsn_pairs')) == 6
    reverse = edges.rename(columns={'i':'j','j':'i','rsn_i':'rsn_j','rsn_j':'rsn_i'})
    pd.testing.assert_frame_equal(model.run(edges,'rsn_pairs'), model.run(reverse,'rsn_pairs'))


def test_partial_network_label_join(edges, tmp_path):
    e = tmp_path/'edges.csv'; n=tmp_path/'nodes.csv'
    edges.drop(columns='rsn_j').to_csv(e,index=False)
    pd.DataFrame({'node_id':range(1,25),'rsn':['Visual','Default','Control']*8}).to_csv(n,index=False)
    actual = load_inputs(e,n)
    assert not actual.rsn_j.isna().any()
    assert 'rsn_i_x' not in actual
    assert len(actual) == len(edges)

@pytest.mark.parametrize('kind,match', [('duplicate','Duplicate'),('self','Self-loops'),('id','positive integer'),('label','missing network'),('column','missing columns')])
def test_invalid_inputs(edges,tmp_path,kind,match):
    E=edges.copy()
    if kind=='duplicate': E=pd.concat([E,E.iloc[[0]].rename(columns={'i':'j','j':'i'})],ignore_index=True)
    if kind=='self': E.loc[0,'j']=E.loc[0,'i']
    if kind=='id': E.loc[0,'i']=0
    if kind=='label': E.loc[0,'rsn_i']=None
    if kind=='column': E=E.drop(columns='FC')
    p=tmp_path/'edges.csv'; E.to_csv(p,index=False)
    with pytest.raises(ValueError,match=match): load_inputs(p)


def test_complete_case_handling_and_empty_response(edges):
    E=edges.copy(); E.loc[0,'FC']=np.inf; E.loc[1,'caliber']=np.nan
    r=model.run(E,'global').iloc[0]
    assert r.n_edges==len(E) and r.n_obs==len(E)-2
    E.FC=np.nan
    assert model.run(E,'global').iloc[0].status=='no_complete_rows'
    E.FC=1
    assert model.run(E,'global').iloc[0].status=='constant_response'


def test_tied_bins(edges):
    E=edges.copy(); E.myelin=1
    assert binning.run_binned(E,'global').empty
    assert binning.bin_correlations(E).empty
    with pytest.raises(ValueError): binning.bin_by_myelin(E,Config(myelin_num_bins=0))

@pytest.mark.parametrize('label',['','  ','.','..'])
def test_invalid_label(label):
    with pytest.raises(ValueError): sanitize_label(label)


def test_cli_all_levels(edges,tmp_path):
    p=tmp_path/'edges.csv'; edges.to_csv(p,index=False)
    result=CliRunner().invoke(main,['--edges',str(p),'--out',str(tmp_path/'out'),'--fc-label','DEMO'])
    assert result.exit_code==0, result.output
    paths=list((tmp_path/'out/DEMO').glob('*.csv'))
    assert len(paths)==8
    assert pd.read_csv(tmp_path/'out/DEMO/nodewise_full.csv').shape[0]==24
    bad=CliRunner().invoke(main,['--edges',str(p),'--out',str(tmp_path/'out'),'--myelin-bins','0'])
    assert bad.exit_code!=0
