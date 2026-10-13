"""Target and context boundaries, including train-only scaling, are explicit."""
import copy

import numpy as np
import pandas as pd
import pytest

from tsflab.catalog.registry.datasets import register_dataset_by_name
from tsflab.data.provider import build_data_loader
from tsflab.experiments.infra.comparison import protocol_fingerprint


@pytest.mark.parametrize('flag,boundary', [('val',24),('test',32)])
@pytest.mark.parametrize('scale', [False,True])
def test_within_split_history_never_crosses_partition_boundary(tmp_path, flag, boundary, scale):
    pd.DataFrame({'date':pd.date_range('2000-01-01',periods=40,freq='h'),'OT':np.arange(40.)}).to_csv(tmp_path/'series.csv',index=False)
    register_dataset_by_name('custom')
    kwargs=dict(dataset_name='custom',root_path=str(tmp_path),data_path='series.csv',size=(3,0,2),
                flag=flag,features='M',dataset_params=dict(target='OT',scale=scale,split_ratio=(.6,.2,.2)),
                batch_size=4,num_workers=0)
    carry,_=build_data_loader(**kwargs)
    independent,loader=build_data_loader(**kwargs,window_context='within-split')
    assert len(carry)==7 and len(independent)==4
    actual_history,actual_target,*_=independent[0]
    expected=np.arange(boundary,boundary+5.)[:,None]
    if scale:
        np.testing.assert_allclose(independent.scaler.mean_,[11.5])
        np.testing.assert_allclose(independent.scaler.mean_,carry.scaler.mean_)
        expected=independent.scaler.transform(expected)
    np.testing.assert_allclose(actual_history,expected[:3],rtol=1e-6)
    np.testing.assert_allclose(actual_target,expected[3:],rtol=1e-6)
    assert len(independent.time_stamp)==8
    assert loader.drop_last is False


def test_new_default_context_preserves_old_protocol_identity():
    old={'task':{'seq_len':3},'evaluation':{},'training':{}}
    new=copy.deepcopy(old);new['task']['window_context']='carry'
    assert protocol_fingerprint(new,{})==protocol_fingerprint(old,{})
    new['task']['window_context']='within-split'
    assert protocol_fingerprint(new,{})!=protocol_fingerprint(old,{})
