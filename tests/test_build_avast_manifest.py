import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    'build_avast_manifest', Path(__file__).parents[1] / 'scripts' / 'build_avast_manifest.py')
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def _write(path, rows):
    with path.open('w') as handle:
        for sha, month, family in rows:
            handle.write(json.dumps({'sample_hash': sha, 'events': [], 'metadata': {'archive_group_id': month},
                                     'family_labels': [family]}) + '\n')


def test_months_are_assigned_whole_and_in_time_order(tmp_path):
    rows = [(f'{i:064x}', f'2019-{1 + i // 10:02d}', 'emotet' if i % 2 else 'zeus') for i in range(100)]
    _write(tmp_path / 'a.jsonl', rows)
    manifest = module.build([tmp_path / 'a.jsonl'])
    parts = manifest['partitions']
    assert sum(p['rows'] for p in parts.values()) == 100
    assert parts['train']['last_month'] < parts['validation']['first_month']
    assert parts['calibration']['last_month'] < parts[module.TEST]['first_month']
    groups = [set(p['archive_groups']) for p in parts.values()]
    assert all(not (a & b) for i, a in enumerate(groups) for b in groups[i + 1:])


def test_duplicates_and_bad_rows_are_rejected(tmp_path):
    _write(tmp_path / 'a.jsonl', [('a' * 64, '2019-01', 'zeus')])
    _write(tmp_path / 'b.jsonl', [('a' * 64, '2019-02', 'zeus')])
    with pytest.raises(ValueError, match='duplicate'):
        module.build([tmp_path / 'a.jsonl', tmp_path / 'b.jsonl'])
    (tmp_path / 'c.jsonl').write_text('{"no": "fields"}\n')
    with pytest.raises(ValueError):
        module.build([tmp_path / 'c.jsonl'])
