"""Verify the shared snapshot and split without requiring the ANN runtime."""
import hashlib
import json
from .data import PROJECT_ROOT, RANDOM_SEED, RAW_FILE, RAW_URL, load_dataset, make_splits
from .preprocessing import RAW_FEATURES, ENGINEERED

REFERENCE = PROJECT_ROOT / 'reports/results/endterm/ann_dataset_manifest.json'

def _hash(value):
    return hashlib.sha256(json.dumps(value, separators=(',', ':'), sort_keys=True).encode()).hexdigest()

def verified_splits():
    df = load_dataset(); splits = make_splits(df, RANDOM_SEED)
    actual = {'source_url': RAW_URL, 'raw_sha256': hashlib.sha256(RAW_FILE.read_bytes()).hexdigest(),
              'n_prepared': len(df), 'split_seed': RANDOM_SEED,
              'split_method': 'existing data.make_splits / GroupShuffleSplit',
              'split_sizes': {name: len(rows) for name, rows in splits.items()},
              'split_row_id_sha256': {name: _hash([int(i) for i in rows.row_id]) for name,rows in splits.items()},
              'split_assignments_sha256': _hash(sorted((int(i),name) for name,rows in splits.items() for i in rows.row_id)),
              'raw_features': RAW_FEATURES, 'feature_set': RAW_FEATURES + ENGINEERED,
              'target': 'mpg', 'target_source':'comb08', 'target_unit':'US MPG'}
    expected=json.loads(REFERENCE.read_text())
    for key, value in actual.items():
        if expected.get(key) != value:
            raise ValueError(f'Endterm dataset/protocol mismatch: {key}')
    return splits, actual
