"""Cheap, training-light ways to raise family top-1 over the TF-IDF + LR baseline.

All variants reuse the same partitions, family vocabulary, targets and metrics as
``classical_family_baseline.py`` and are scored on validation only:

- ``lr``: TF-IDF + one-vs-rest logistic regression on a feature set and C;
- feature sets add event trigrams (``tri``) and argument-free event tokens
  (``reduced``: category:api:resource) to the baseline tokens;
- ``knn``: cosine k-nearest neighbours on the same TF-IDF vectors;
- ``rank``: rank-averaged fusion of an LR and a kNN variant.

Because many variants are compared on one validation set, the reported estimate
for "pick the best variant" is group cross-fitted: the variant is chosen on half
of the archive groups and scored on the other half, then the halves swap.
Withheld test partitions are never parsed.
"""
from __future__ import annotations
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.stats import rankdata
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from train_winmet_expansion import load_expansion  # noqa: E402

_spec = importlib.util.spec_from_file_location('classical_family_baseline', ROOT / 'classical_family_baseline.py')
classical = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(classical)

FEATURE_SETS = {
    'base': ('e', 'b', 's'),
    'base+tri': ('e', 'b', 't', 's'),
    'base+reduced': ('e', 'b', 'r', 'rb', 's'),
    'base+tri+reduced': ('e', 'b', 't', 'r', 'rb', 's'),
}


def report_token_groups(report):
    events = [event.token for event in report.events]
    reduced = [':'.join(token.split(':')[:3]) for token in events]
    groups = {
        'e': [f'e:{t}' for t in events],
        'b': [f'b:{a}>{b}' for a, b in zip(events, events[1:])],
        't': [f't:{a}>{b}>{c}' for a, b, c in zip(events, events[1:], events[2:])],
        'r': [f'r:{t}' for t in reduced],
        'rb': [f'rb:{a}>{b}' for a, b in zip(reduced, reduced[1:])],
        's': classical.report_tokens(report)['evidence'],
    }
    return groups


def documents(groups_list, feature_set):
    keys = FEATURE_SETS[feature_set]
    return [[token for key in keys for token in groups[key]] for groups in groups_list]


def vectorize(train_docs, valid_docs):
    vectorizer = TfidfVectorizer(analyzer=lambda doc: doc, lowercase=False, min_df=2,
                                 sublinear_tf=True, max_features=400_000, dtype=np.float32)
    return vectorizer.fit_transform(train_docs), vectorizer.transform(valid_docs)


def knn_scores(x_train, train_y, x_valid, k):
    """Similarity-weighted votes of the k most cosine-similar train rows (rows are L2-normalised)."""
    scores = np.zeros((x_valid.shape[0], train_y.shape[1]))
    for start in range(0, x_valid.shape[0], 256):
        sims = (x_valid[start:start + 256] @ x_train.T).toarray()
        top = np.argpartition(-sims, kth=min(k, sims.shape[1] - 1), axis=1)[:, :k]
        for row, neighbours in enumerate(top):
            weights = sims[row, neighbours]
            scores[start + row] = weights @ train_y[neighbours] / max(weights.sum(), 1e-12)
    return scores


def rank_fuse(a, b):
    return (rankdata(a, axis=1) + rankdata(b, axis=1)) / 2.0


def top1_rows(targets, scores, mask):
    """Per-row top-1 correctness (NaN where the row has no in-vocabulary family)."""
    known = (mask > 0) & (targets >= 0.5)
    rows = known.any(axis=1)
    top = np.argmax(scores, axis=1)
    out = np.full(len(targets), np.nan)
    out[rows] = known[rows, top[rows]]
    return out


def cross_fitted_selection(per_variant_correct, groups):
    """Choose the best variant on one half of the groups, score on the other half, then swap."""
    ordered = sorted(set(groups))
    fold = np.asarray([ordered.index(g) % 2 for g in groups])
    names = list(per_variant_correct)
    chosen, correct = {}, []
    for held in (0, 1):
        fit = fold != held
        best = max(names, key=lambda n: np.nanmean(per_variant_correct[n][fit]))
        chosen[held] = best
        values = per_variant_correct[best][fold == held]
        correct.extend(values[~np.isnan(values)])
    return {'top1': float(np.mean(correct)), 'fold_choices': chosen, 'archive_groups': len(ordered)}


def run(train, valid, *, cs=(16.0, 32.0, 64.0), ks=(5, 15), seed=44, log=print):
    labels = classical.family_vocab(train, 5)
    train_y, train_mask = classical.family_targets(train, labels)
    valid_y, valid_mask = classical.family_targets(valid, labels)
    keep = train_mask[:, 0] > 0
    priors = {label: float(train_y[keep, i].mean()) for i, label in enumerate(labels)}
    train_groups = [report_token_groups(r) for r, k in zip(train, keep) if k]
    valid_groups = [report_token_groups(r) for r in valid]
    y = train_y[keep]
    scores, results = {}, {}

    def record(name, value, started):
        scores[name] = value
        metrics = classical.evaluate(valid_y, value, valid_mask, labels, priors)
        results[name] = {k: metrics[k] for k in ('family_top1_accuracy', 'family_top3_accuracy',
                                                 'family_map', 'family_macro_f1', 'family_evaluated_samples')}
        results[name]['seconds'] = round(time.perf_counter() - started, 1)
        log(json.dumps({'variant': name, **results[name]}))

    for feature_set in FEATURE_SETS:
        x_train, x_valid = vectorize(documents(train_groups, feature_set), documents(valid_groups, feature_set))
        for c in cs:
            started = time.perf_counter()
            model = classical.OneVsRestClassifier(classical.LogisticRegression(
                C=c, solver='liblinear', max_iter=2000, random_state=seed))
            model.fit(x_train, y)
            record(f'lr|{feature_set}|C={c:g}', model.predict_proba(x_valid), started)
        for k in ks:
            started = time.perf_counter()
            record(f'knn|{feature_set}|k={k}', knn_scores(x_train, y, x_valid, k), started)
    lr_names = [n for n in scores if n.startswith('lr|')]
    knn_names = [n for n in scores if n.startswith('knn|')]
    best_lr = max(lr_names, key=lambda n: results[n]['family_map'] or -1)
    for knn in knn_names:
        started = time.perf_counter()
        record(f'rank|{best_lr}+{knn}', rank_fuse(scores[best_lr], scores[knn]), started)
    correct = {name: top1_rows(valid_y, value, valid_mask) for name, value in scores.items()}
    groups = [r.metadata.get('archive_group_id') for r in valid]
    baseline = f'lr|base|C={16.0 if 16.0 in cs else cs[0]:g}'  # the selected classical config
    return {
        'labels': len(labels), 'variants': results, 'baseline_variant': baseline,
        'best_on_validation': max(results, key=lambda n: results[n]['family_top1_accuracy'] or -1),
        'cross_fitted_best_variant': cross_fitted_selection(correct, groups),
        'cross_fitted_baseline_only': cross_fitted_selection({baseline: correct[baseline]}, groups),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('refusing to overwrite output')
    manifest = json.loads(args.manifest.read_text())
    reports = load_expansion(manifest.get('source'), manifest, pack=True)  # no withheld partitions
    result = run(reports['train'], reports['validation'], log=lambda line: print(line, flush=True))
    result['notice'] = ('Validation only. The cross-fitted figure is the honest estimate for choosing the best '
                        'variant; per-variant validation numbers are optimistic when compared with each other.')
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: result[k] for k in ('best_on_validation', 'cross_fitted_best_variant',
                                             'cross_fitted_baseline_only')}), flush=True)


if __name__ == '__main__':
    main()
