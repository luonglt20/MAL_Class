import importlib.util
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip('sklearn')

SPEC = importlib.util.spec_from_file_location(
    'fuse_attention_tfidf', Path(__file__).parents[1] / 'scripts' / 'fuse_attention_tfidf.py')
fusion = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fusion)

LABELS = ['alpha', 'beta']
PRIORS = {'alpha': .5, 'beta': .5}


def _data(n=40, seed=0):
    rng = np.random.default_rng(seed)
    truth = rng.integers(0, 2, n)
    targets = np.eye(2)[truth]
    mask = np.ones_like(targets)
    good = targets * .8 + rng.random((n, 2)) * .3
    noise = rng.random((n, 2))
    return targets, mask, good, noise


def test_fusion_weight_endpoints_are_single_models():
    att, tf = np.full((2, 2), .9), np.full((2, 2), .1)
    assert np.allclose(fusion.fuse(att, tf, 1.0), att)
    assert np.allclose(fusion.fuse(att, tf, 0.0), tf)


def test_best_weight_prefers_informative_scores():
    targets, mask, good, noise = _data()
    weight, results = fusion.best_weight(good, noise, targets, mask, LABELS, PRIORS)
    # Ties at perfect mAP resolve to the smallest tied weight, so only the side is asserted.
    assert weight > .5
    assert results[str(weight)]['family_top1_accuracy'] == 1.0
    weight, _ = fusion.best_weight(noise, good, targets, mask, LABELS, PRIORS)
    assert weight < .5


def test_cross_fitting_chooses_weights_without_the_scored_fold():
    targets, mask, good, noise = _data()
    groups = [f'g{i % 4}' for i in range(len(targets))]
    metrics, chosen, n_groups = fusion.cross_fitted(good, noise, targets, mask, groups, LABELS, PRIORS)
    assert n_groups == 4 and set(chosen) == {0, 1}
    assert metrics['family_top1_accuracy'] == 1.0
