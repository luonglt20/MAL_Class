import importlib.util
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip('sklearn')

from malware_hybrid.schema import ExecutionEvent, ParsedReport

SPEC = importlib.util.spec_from_file_location(
    'behavior_signature_baseline', Path(__file__).parents[1] / 'scripts' / 'behavior_signature_baseline.py')
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def _report(name, apis, signatures):
    events = [ExecutionEvent(i, float(i), '1', '0', '1', api, 'process', True, '0', 'p', 't', 'file')
              for i, api in enumerate(apis)]
    return ParsedReport(name, family=None, atomic_evidence=[], events=events,
                        metadata={'cape_signatures': signatures})


def _corpus(prefix, n):
    # "order" depends on A before B; both orders contain the same unigrams.
    rows = []
    for i in range(n):
        rows.append(_report(f'{prefix}o{i}', ['A', 'B', f'X{i % 3}'], ['order', 'common']))
        rows.append(_report(f'{prefix}r{i}', ['B', 'A', f'X{i % 3}'], ['common']))
    rows.append(_report(f'{prefix}none', ['A'], []))
    return rows


def test_targets_mask_reports_without_signatures():
    reports = _corpus('t', 2)
    labels = module.signature_vocab(reports, 2)
    assert labels == ['common', 'order']
    targets, mask = module.signature_targets(reports, labels)
    assert mask[-1].sum() == 0 and mask[0].all()
    assert targets[0].tolist() == [1.0, 1.0] and targets[1].tolist() == [1.0, 0.0]


def test_unigrams_drop_order_while_bigrams_keep_it():
    report = _report('a', ['A', 'B'], ['x'])
    assert all(t.startswith('e:') for t in module.event_tokens(report, 'unigram'))
    assert any(t.startswith('b:') for t in module.event_tokens(report, 'bigram'))


def test_bigram_encoding_learns_an_order_only_behavior():
    result = module.run(_corpus('t', 8), _corpus('v', 4), min_support=4, log=lambda _: None)
    ap = {enc: result['encodings'][enc]['validation']['per_label_ap']['order'] for enc in module.ENCODINGS}
    assert ap['bigram'] == pytest.approx(1.0)
    assert ap['unigram'] < 1.0
    assert result['bigram_minus_unigram']['labels_better'] >= 1


def test_paired_bootstrap_is_zero_for_identical_scores():
    targets = np.array([[1, 0], [0, 1], [1, 1]], dtype=float)
    scores = np.array([[.9, .1], [.2, .8], [.7, .6]])
    out = module.bootstrap_map_difference(targets, scores, scores, np.ones_like(targets), rounds=10)
    assert out['mean'] == 0.0
