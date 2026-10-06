"""Load MATLAB subject stacks and audit participant overlap before modeling."""
from pathlib import Path
import json
import re
import numpy as np
import pandas as pd
from scipy.io import loadmat


def participant_id(scan_id):
    """Canonicalize this project's sub-NN / sub-NNr convention only."""
    value = str(scan_id).strip()
    if not re.fullmatch(r"sub-\d+r?", value):
        raise ValueError(f"Unsupported scan ID {value!r}; provide an explicit participant ID")
    return value.removesuffix("r")


def _read_entry(root, entry):
    path = root / entry["path"]
    try:
        data = loadmat(path, simplify_cells=True)
    except NotImplementedError as exc:
        raise ValueError("MATLAB v7.3 files need conversion to -v7 or an HDF5 adapter") from exc
    key = entry.get("key", "Dts")
    if key not in data:
        raise ValueError(f"{path.name}: missing variable {key!r}")
    return np.asarray(data[key])


def load_subject_manifest(path):
    """Read explicit file/key mappings; never infer stack order from filenames."""
    path = Path(path)
    spec = json.loads(path.read_text())
    batches = {}
    for name, batch in spec["batches"].items():
        scans = [str(x).strip() for x in _read_entry(path.parent, batch["ids"]).reshape(-1)]
        if len(scans) != len(set(scans)):
            raise ValueError(f"{name}: duplicate scan IDs")
        people = batch.get("participant_ids")
        if people is None:
            people = [participant_id(x) for x in scans]
        if len(people) != len(scans) or any(not str(x).strip() for x in people):
            raise ValueError(f"{name}: invalid participant IDs")
        arrays = {}
        for feature in ("caliber", "myelin", "length", "FC"):
            a = _read_entry(path.parent, batch[feature]).astype(float)
            if a.ndim == 2 and len(scans) == 1:
                a = a[:, :, None]
            if a.ndim != 3 or a.shape[0] != a.shape[1] or a.shape[2] != len(scans):
                raise ValueError(f"{name}/{feature}: expected nodes x nodes x {len(scans)}, got {a.shape}")
            if not np.allclose(a, a.transpose(1, 0, 2), equal_nan=True):
                raise ValueError(f"{name}/{feature}: matrices are not symmetric")
            arrays[feature] = a
        if len({a.shape for a in arrays.values()}) != 1:
            raise ValueError(f"{name}: feature shapes disagree")
        batches[name] = dict(scan_ids=scans, participant_ids=list(map(str, people)), arrays=arrays)
    if len({b['arrays']['FC'].shape[0] for b in batches.values()}) != 1:
        raise ValueError("Batches have different node counts")
    return batches


def training_indices(batch, held_out_person):
    """Exclude every session of a participant, including a retained rescan."""
    return np.flatnonzero(np.array(batch['participant_ids']) != held_out_person)


def audit_subjects(batches):
    rows = []
    for name, batch in batches.items():
        for k, scan in enumerate(batch['scan_ids']):
            row = dict(batch=name, scan_id=scan, participant_id=batch['participant_ids'][k])
            for feature, stack in batch['arrays'].items():
                values = stack[:, :, k][np.triu_indices(stack.shape[0], 1)]
                row[feature + '_nonfinite'] = int((~np.isfinite(values)).sum())
                row[feature + '_zero'] = int((values == 0).sum())
            rows.append(row)
    return pd.DataFrame(rows)


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    batches = load_subject_manifest(args.manifest)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    audit_subjects(batches).to_csv(out / 'subject_audit.csv', index=False)
    if 'main' in batches and 'holdout' in batches:
        rows = []
        for person in dict.fromkeys(batches['holdout']['participant_ids']):
            keep = training_indices(batches['main'], person)
            excluded = sorted(set(range(len(batches['main']['scan_ids']))) - set(keep))
            rows.append(dict(participant_id=person, n_training=len(keep), excluded_scans=';'.join(batches['main']['scan_ids'][i] for i in excluded)))
        pd.DataFrame(rows).to_csv(out / 'fold_exclusions.csv', index=False)
    print(f'Audit saved to {out}')


if __name__ == '__main__':
    main()
