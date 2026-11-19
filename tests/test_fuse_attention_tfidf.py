import importlib.util
from pathlib import Path
from types import SimpleNamespace

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


@pytest.mark.parametrize('features', ['all', 'events', 'evidence'])
def test_tfidf_streams_tokens_and_preserves_scores_and_group_order(monkeypatch, features):
    def report(label, supervised=True):
        return SimpleNamespace(families=[label], family_supervised=supervised)

    train = [report(label) for label in LABELS for _ in range(4)]
    train.append(report('ignored', supervised=False))
    groups = [[report('beta'), report('alpha')], [report('alpha')]]
    calls = []

    def tokens(row):
        calls.append(row)
        return {'events': ['event:' + row.families[0]],
                'evidence': ['static:' + row.families[0]]}

    monkeypatch.setattr(fusion.classical, 'report_tokens', tokens)
    pick = (lambda t: t['events'] + t['evidence']) if features == 'all' else (lambda t: t[features])
    targets, _ = fusion.classical.family_targets(train[:-1], LABELS)
    expected, _ = fusion.classical.fit_predict(
        [pick(tokens(r)) for r in train[:-1]], targets,
        [pick(tokens(r)) for group in groups for r in group], 4.0, 44)
    calls.clear()
    original_fit = fusion.classical.fit_predict

    def streaming_fit(train_docs, train_y, valid_docs, c, seed):
        assert calls == []  # Tokenization must wait for vectorizer consumption.
        assert iter(train_docs) is train_docs
        assert iter(valid_docs) is valid_docs
        return original_fit(train_docs, train_y, valid_docs, c, seed)

    monkeypatch.setattr(fusion.classical, 'fit_predict', streaming_fit)
    scores, priors = fusion.tfidf_scores(train, groups, LABELS, features, 4.0, 44)
    np.testing.assert_array_equal(scores[0], expected[:2])
    np.testing.assert_array_equal(scores[1], expected[2:])
    assert priors == PRIORS
    assert len(calls) == 11
    assert all(row.family_supervised for row in calls)
