import importlib.util
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip('sklearn')

from malware_hybrid.schema import AtomicEvidence, ExecutionEvent, ParsedReport

SPEC = importlib.util.spec_from_file_location(
    'family_cheap_boosts', Path(__file__).parents[1] / 'scripts' / 'family_cheap_boosts.py')
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def _report(name, family, apis, group):
    events = [ExecutionEvent(i, float(i), '1', '0', '1', api, 'process', True, '0', 'p', 't', 'file',
                             arguments={'path': f'{name}-{i}'})
              for i, api in enumerate(apis)]
    return ParsedReport(name, family=family, events=events, atomic_evidence=[AtomicEvidence('file_size', 'x')],
                        metadata={'family_supervised': True, 'archive_group_id': group})


def _corpus(prefix, n):
    rows = []
    for i in range(n):
        g = f'{prefix}{i % 4}'
        rows.append(_report(f'{prefix}a{i}', 'alpha', ['CreateFileW', 'WriteFile', 'CloseHandle'], g))
        rows.append(_report(f'{prefix}b{i}', 'beta', ['RegOpenKeyExW', 'RegSetValueExW', 'CloseHandle'], g))
    return rows


def test_reduced_tokens_drop_arguments_and_trigrams_need_three_events():
    groups = module.report_token_groups(_report('x', 'alpha', ['A', 'B', 'C'], 'g'))
    assert len(groups['t']) == 1 and len(groups['b']) == 2
    assert all(token.count(':') == 3 for token in groups['r'])  # 'r:' + category:api:resource
    assert all('path=' not in token for token in groups['r'])


def test_knn_votes_follow_the_nearest_neighbour():
    from scipy.sparse import csr_matrix
    x_train = csr_matrix(np.array([[1.0, 0.0], [0.0, 1.0]]))
    y = np.array([[1.0, 0.0], [0.0, 1.0]])
    scores = module.knn_scores(x_train, y, csr_matrix(np.array([[0.9, 0.1]])), k=1)
    assert scores.argmax() == 0


def test_cross_fitted_selection_scores_each_half_with_the_other_halfs_choice():
    groups = ['g0', 'g1', 'g0', 'g1']
    correct = {'good': np.array([1.0, 1.0, 1.0, 1.0]), 'bad': np.array([0.0, 0.0, 0.0, np.nan])}
    out = module.cross_fitted_selection(correct, groups)
    assert out['top1'] == 1.0 and set(out['fold_choices'].values()) == {'good'}


def test_argument_free_tokens_generalise_when_arguments_are_sample_specific():
    # Every event carries a per-sample path, so full tokens never repeat across reports.
    result = module.run(_corpus('t', 8), _corpus('v', 4), cs=(4.0,), ks=(3,), log=lambda _: None)
    assert result['variants']['lr|base|C=4']['family_top1_accuracy'] == 0.5
    assert result['variants']['lr|base+reduced|C=4']['family_top1_accuracy'] == 1.0
    assert result['cross_fitted_best_variant']['top1'] == 1.0
