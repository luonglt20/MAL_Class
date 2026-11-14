import importlib.util
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip('sklearn')

from malware_hybrid.schema import ParsedReport

SPEC = importlib.util.spec_from_file_location(
    'tlsh_routed_hybrid', Path(__file__).parents[1] / 'scripts' / 'tlsh_routed_hybrid.py')
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def _report(name, family, digest):
    return ParsedReport(name, family=family, atomic_evidence=[], events=[],
                        metadata={'family_supervised': True, 'fuzzy_hashes': {'tlsh': digest}})


def test_nearest_neighbour_uses_only_labelled_in_vocabulary_train_reports():
    train = [_report('a', 'alpha', '10'), _report('b', 'beta', '50'), _report('c', 'gamma', '11')]
    valid = [_report('v', 'alpha', '12'), _report('w', 'alpha', '')]
    distance = lambda x, y: abs(int(x) - int(y))
    out = module.nearest_neighbours(train, valid, ['alpha', 'beta'], distance)
    assert out[0] == (2, 0)  # gamma is out of vocabulary and ignored
    assert out[1] == (float('inf'), -1)


def test_hybrid_routes_only_below_the_threshold():
    lr = np.array([[0.1, 0.9], [0.9, 0.1]])
    known = np.array([[True, False], [True, False]])
    neighbours = [(5, 0), (80, 1)]
    assert module.hybrid_correct(lr, neighbours, known, 10).tolist() == [1.0, 1.0]
    assert module.hybrid_correct(lr, neighbours, known, -1).tolist() == [0.0, 1.0]


def test_cross_fitting_reports_hybrid_and_lr_on_the_same_rows():
    lr = np.array([[0.1, 0.9]] * 4)
    known = np.array([[True, False]] * 4)
    neighbours = [(5, 0)] * 4
    out = module.cross_fitted_threshold(lr, neighbours, known, ['g0', 'g1', 'g0', 'g1'], thresholds=(0, 10))
    assert out['hybrid_top1'] == 1.0 and out['lr_top1_same_rows'] == 0.0 and out['rows'] == 4
