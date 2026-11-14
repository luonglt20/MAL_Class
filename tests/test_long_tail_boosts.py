import importlib.util
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip('sklearn')

SPEC = importlib.util.spec_from_file_location(
    'long_tail_boosts', Path(__file__).parents[1] / 'scripts' / 'long_tail_boosts.py')
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def test_logit_adjustment_lifts_rare_families():
    scores = np.array([[0.6, 0.5]])
    priors = np.array([0.9, 0.1])
    assert np.argmax(scores) == 0
    assert np.argmax(module.logit_adjust(scores, priors, 1.0)) == 1
    assert np.allclose(module.logit_adjust(scores, priors, 0.0), scores)


def test_mcnemar_counts_only_discordant_rows():
    base = np.array([1, 0, 0, 1, np.nan])
    cand = np.array([1, 1, 1, 0, 1])
    out = module.mcnemar(base, cand)
    assert (out['gained'], out['lost']) == (2, 1)
    assert 0 < out['p_value_exact'] <= 1


def test_cross_fitting_picks_on_other_half():
    correct = {'a': np.array([1.0, 0.0, 1.0, 0.0]), 'b': np.array([0.0, 1.0, 0.0, 1.0])}
    combined, chosen = module.cross_fitted(correct, ['g0', 'g1', 'g0', 'g1'])
    # fold 0 = g0 rows, chosen on g1 rows where 'b' wins, so g0 rows get b's (wrong) answers
    assert chosen == {0: 'b', 1: 'a'}
    assert np.nanmean(combined) == 0.0
