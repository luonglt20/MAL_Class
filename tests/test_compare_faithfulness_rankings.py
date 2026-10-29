import copy
import importlib.util
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location('compare_rankings', Path(__file__).parents[1] / 'scripts/compare_faithfulness_rankings.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def report(ranking, score):
    good = {'evaluated': True, 'valid_tokens': 10, 'masked_counts': [1, 2], 'random_aopc': .1,
            'random_mean_confidence_drop': [.08, .12],
            'top_minus_random_aopc': score}
    skipped = {'evaluated': False}
    sample = {'sample_hash': 'identity', 'target_index': 1, 'seed': 44,
              'fractions': [.1, .2], 'random_repeats': 5, 'baseline_probability': .6,
              'branches': {'signature': good, 'static': copy.deepcopy(good),
                           'temporal_events': skipped, 'graph': skipped}}
    return {'source_sha256': 'source', 'checkpoint_sha256': 'checkpoint',
            'evaluated_partition': 'validation', 'static_feature_faithfulness': {
                'intervention': 'static_feature', 'sample_ids_sha256': 'fingerprint',
                'ranking': ranking, 'samples': [sample]}}


def test_comparison_requires_matched_controls_and_reports_paired_change():
    left, right = report('attention', -.2), report('signed_readout', .1)
    result = module.compare(left, right)
    assert result['branches']['signature']['mean_right_minus_left_aopc'] == pytest.approx(.3)
    assert result['branches']['signature']['right_better'] == 1
    assert result['branches']['graph']['paired_samples'] == 0
    right['static_feature_faithfulness']['samples'][0]['branches']['signature']['random_aopc'] = .2
    with pytest.raises(ValueError, match='random controls differ'):
        module.compare(left, right)


def test_comparison_rejects_different_sample_or_checkpoint():
    left, right = report('attention', 0), report('signed_readout', 0)
    right['checkpoint_sha256'] = 'different'
    with pytest.raises(ValueError, match='checkpoint_sha256'):
        module.compare(left, right)
    right = report('signed_readout', 0)
    right['static_feature_faithfulness']['samples'][0]['sample_hash'] = 'other'
    with pytest.raises(ValueError, match='paired sample'):
        module.compare(left, right)
