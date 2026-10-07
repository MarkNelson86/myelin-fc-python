import pandas as pd
import pytest
from myelinfccoupling.compare_runs import compare


def fixture(tmp_path):
    frame=pd.DataFrame({'participant_id':['p']*4,'scan_id':['s']*4,'i':[1,1,1,2],'j':[2,3,4,3],
        'empirical':[.1,.2,.4,.5],'template':[.1,.3,.3,.4],
        'OLS':[.1,.25,.35,.5],'Ridge':[.1,.2,.4,.5]})
    a=tmp_path/'a.csv';b=tmp_path/'b.csv';frame.to_csv(a,index=False);frame.iloc[1:].to_csv(b,index=False)
    return frame,a,b


def test_comparison_intersects_edges_and_scores_against_template(tmp_path):
    frame,a,b=fixture(tmp_path);out=tmp_path/'comparison'
    summary=compare({'A':a,'B':b},out)
    coverage=pd.read_csv(out/'edge_coverage.csv')
    assert coverage.common_edges.tolist()==[3,3]
    assert summary.loc['Template','R2_vs_template']==0
    assert summary.loc['A / Ridge','R2_vs_template']==1
    assert summary.loc['A / OLS','rmse']==summary.loc['B / OLS','rmse']


def test_comparison_rejects_mismatched_targets_and_duplicate_keys(tmp_path):
    frame,a,b=fixture(tmp_path);frame.loc[1,'empirical']+=1;frame.to_csv(b,index=False)
    with pytest.raises(ValueError,match='differs'):compare({'A':a,'B':b},tmp_path/'out')
    pd.concat([frame,frame.iloc[:1]]).to_csv(b,index=False)
    with pytest.raises(ValueError,match='duplicate'):compare({'A':a,'B':b},tmp_path/'out')
