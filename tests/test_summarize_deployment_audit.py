import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('audit_summary', Path(__file__).parents[1] / 'scripts/summarize_deployment_audit.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_rejects_missing_or_inconsistent_rows():
    with pytest.raises(ValueError, match='no saved'):
        module.summarize({}, 'sha')
    with pytest.raises(ValueError, match='sample count'):
        module.summarize({'deployment_family_decisions': {'samples': 2, 'rows': []}}, 'sha')


def test_summarizes_labeled_rows_without_sample_ids():
    row = {'sample_hash': 'not-for-output', 'reference_families': ['family-a'],
           'reference_known_family': True, 'accepted': True, 'primary_matches_reference': True}
    document = {'checkpoint_sha256': 'model', 'evaluated_partition': 'validation',
                'deployment_family_decisions': {'samples': 1, 'errors': 0, 'rows': [row]}}
    result = module.summarize(document, 'sha')
    assert result['summary']['all_reference_labeled']['correct_fraction_all_labeled'] == 1
    assert 'not-for-output' not in str(result)
