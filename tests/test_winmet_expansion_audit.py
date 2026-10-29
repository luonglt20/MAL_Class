import importlib.util
from pathlib import Path
import random

import pytest

from malware_hybrid.schema import ParsedReport
from malware_hybrid.training import research_split

spec = importlib.util.spec_from_file_location('expansion', Path(__file__).parents[1] / 'scripts/audit_winmet_expansion.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def cohort():
    rows = [ParsedReport(sample_hash=str(i), family=f'family-{i // 10}', atomic_evidence=[], events=[],
                         metadata={'archive_group_id': f'block-{i // 10}'}) for i in range(100)]
    selected, rng = rows[:90], random.Random(44)
    for index in range(90, len(rows)):
        replacement = rng.randrange(index + 1)
        if replacement < 90:
            selected[replacement] = rows[index]
    parts = research_split(selected, 44)
    reference = {'sample_seed': 44, 'split_seed': 44, 'sample_cap': 90,
                 'split_strategy': parts[-1], 'partitions': {
                     name: {'sample_ids_sha256': module.fingerprint(part)}
                     for name,part in zip(module.NAMES, parts[:4])}}
    return rows, reference, parts


def test_expansion_retains_old_groups_and_only_counts_training_families():
    rows, reference, parts = cohort()
    result = module.audit(rows, reference)
    train_groups = {r.metadata['archive_group_id'] for r in parts[0]}
    expected = {r.family for r in rows if r.metadata['archive_group_id'] in train_groups}
    assert set(result['train_family_support']) == expected
    assert all(value == 10 for value in result['train_family_support'].values())
    assert sum(p['rows'] for p in result['partitions'].values()) == 100
    assert result['old_group_assignments_preserved']


def test_expansion_rejects_mismatched_original_partition():
    rows, reference, _ = cohort()
    reference['partitions']['train']['sample_ids_sha256'] = 'invalid'
    with pytest.raises(ValueError, match='fingerprint'):
        module.audit(rows, reference)
