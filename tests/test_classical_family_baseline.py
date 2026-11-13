import importlib.util
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip('sklearn')

from malware_hybrid.schema import AtomicEvidence, ExecutionEvent, ParsedReport

SPEC = importlib.util.spec_from_file_location(
    'classical_family_baseline', Path(__file__).parents[1] / 'scripts' / 'classical_family_baseline.py')
baseline = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(baseline)


def _report(name, family, apis, supervised=True):
    events = [ExecutionEvent(i, float(i), '1', '0', '1', api, 'process', True, '0', 'p', 't', 'file')
              for i, api in enumerate(apis)]
    return ParsedReport(name, family=family, events=events,
                        atomic_evidence=[AtomicEvidence('file_size', family or 'none')],
                        metadata={'family_supervised': supervised})


def _corpus(prefix, n):
    reports = []
    for i in range(n):
        reports.append(_report(f'{prefix}a{i}', 'alpha', ['CreateFileW', 'WriteFile', f'X{i % 3}']))
        reports.append(_report(f'{prefix}b{i}', 'beta', ['RegSetValueExW', 'CreateProcessW', f'Y{i % 3}']))
    return reports


def test_ranking_accuracy_matches_trainer_definition():
    targets = np.array([[1, 0, 0], [0, 1, 1], [0, 0, 0]], dtype=float)
    mask = np.array([[1, 1, 1], [1, 1, 1], [0, 0, 0]], dtype=float)
    scores = np.array([[.9, .1, 0], [.5, .4, .1], [1, 0, 0]])
    top1, top3, n = baseline.ranking_accuracy(targets, scores, mask)
    assert (top1, top3, n) == (0.5, 1.0, 2)


def test_vocab_and_targets_respect_support_and_supervision():
    reports = _corpus('t', 5) + [_report('rare', 'gamma', ['A']), _report('u', 'alpha', ['A'], supervised=False)]
    labels = baseline.family_vocab(reports, min_support=5)
    assert labels == ['alpha', 'beta']
    targets, mask = baseline.family_targets(reports[-2:], labels)
    assert targets.sum() == 0 and mask[0].all() and not mask[1].any()


def test_separable_families_are_learned():
    result = baseline.run(_corpus('t', 6), _corpus('v', 3), cs=(4.0,), feature_sets=('events', 'all'),
                          log=lambda _: None)
    assert result['labels'] == ['alpha', 'beta']
    for entry in result['runs']:
        assert entry['validation']['family_top1_accuracy'] == 1.0
        assert entry['validation']['family_evaluated_samples'] == 6
