import json
import numpy as np
import pytest
from scipy.io import savemat
from myelinfccoupling.subjects import participant_id, training_indices, load_subject_manifest, audit_subjects


def test_rescan_exclusion():
    batch = {'participant_ids': ['sub-22', 'sub-25', 'sub-29'], 'scan_ids': ['sub-22', 'sub-25r', 'sub-29']}
    assert participant_id('sub-25r') == 'sub-25'
    assert training_indices(batch, 'sub-25').tolist() == [0, 2]
    with pytest.raises(ValueError):
        participant_id('unrecognized')


def test_mat_roundtrip_and_misalignment(tmp_path):
    savemat(tmp_path / 'ids.mat', {'Ss': np.array(['sub-02', 'sub-25r'], dtype=object)})
    a = np.zeros((4, 4, 2)); a[0, 1, :] = a[1, 0, :] = [1, 2]
    savemat(tmp_path / 'data.mat', {'Dts': a})
    batch = {'ids': {'path': 'ids.mat', 'key': 'Ss'}}
    batch.update({k: {'path': 'data.mat'} for k in ['caliber', 'myelin', 'length', 'FC']})
    path = tmp_path / 'manifest.json'
    path.write_text(json.dumps({'batches': {'main': batch}}))
    loaded = load_subject_manifest(path)
    assert loaded['main']['participant_ids'] == ['sub-02', 'sub-25']
    assert len(audit_subjects(loaded)) == 2
    savemat(tmp_path / 'data.mat', {'Dts': a[:, :, :1]})
    with pytest.raises(ValueError, match='expected nodes'):
        load_subject_manifest(path)
