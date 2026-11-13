"""Classical TF-IDF + one-vs-rest logistic regression family baseline.

Uses the same frozen expansion partitions, family vocabulary (train support >= 5),
targets, masks and ranking metrics as the Hybrid Attention trainer, so validation
numbers are directly comparable. Withheld test partitions are never parsed.
Model selection (C, feature set) uses validation family mAP only.
"""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

from malware_hybrid.metrics import family_prior_baseline, multilabel_f1, multilabel_metrics
from malware_hybrid.signature_policy import model_safe_evidence

FEATURE_SETS = ('events', 'evidence', 'all')


def report_tokens(report, max_events=None):
    """Bag of event tokens, event bigrams and model-safe static/signature evidence."""
    events = [event.token for event in report.events[:max_events]]
    dynamic = [f'e:{token}' for token in events]
    dynamic += [f'b:{left}>{right}' for left, right in zip(events, events[1:])]
    evidence = [f's:{item.token}' for item in model_safe_evidence(report.atomic_evidence)]
    return {'events': dynamic, 'evidence': evidence}


def family_vocab(reports, min_support=5):
    counts = Counter()
    for report in reports:
        if report.family_supervised:
            counts.update(report.families)
    return sorted(family for family, count in counts.items() if count >= min_support)


def family_targets(reports, labels):
    index = {label: i for i, label in enumerate(labels)}
    targets = np.zeros((len(reports), len(labels)))
    mask = np.zeros_like(targets)
    for row, report in enumerate(reports):
        if report.family_supervised:
            mask[row] = 1.0
            for family in report.families:
                if family in index:
                    targets[row, index[family]] = 1.0
    return targets, mask


def ranking_accuracy(targets, scores, mask):
    """Top-1/top-3 over rows with a known in-vocabulary family, as in training.evaluate."""
    known = (mask > 0) & (targets >= 0.5)
    rows = known.any(axis=1)
    if not rows.any():
        return None, None, 0
    top = np.argsort(-scores[rows], axis=1, kind='stable')
    positives = known[rows]
    top1 = float(positives[np.arange(len(top)), top[:, 0]].mean())
    top3 = float(np.any(np.take_along_axis(positives, top[:, :3], axis=1), axis=1).mean())
    return top1, top3, int(rows.sum())


def evaluate(targets, scores, mask, labels, priors):
    top1, top3, n = ranking_accuracy(targets, scores, mask)
    ranking = multilabel_metrics(targets, scores, mask, labels, prefix='family')
    f1 = multilabel_f1(targets, scores, mask)
    return {
        'family_map': ranking['family_map'], 'family_macro_f1': f1['macro_f1'],
        'family_top1_accuracy': top1, 'family_top3_accuracy': top3, 'family_evaluated_samples': n,
        **family_prior_baseline(targets, mask, labels, priors),
    }


def fit_predict(train_docs, train_y, valid_docs, c, seed):
    vectorizer = TfidfVectorizer(analyzer=lambda doc: doc, lowercase=False, min_df=2,
                                 sublinear_tf=True, max_features=300_000, dtype=np.float32)
    x_train = vectorizer.fit_transform(train_docs)
    x_valid = vectorizer.transform(valid_docs)
    model = OneVsRestClassifier(
        # Sequential: joblib memmaps large sparse inputs read-only, which liblinear rejects.
        LogisticRegression(C=c, solver='liblinear', max_iter=2000, random_state=seed))
    model.fit(x_train, train_y)
    return model.predict_proba(x_valid), x_train.shape[1]


def run(train_reports, valid_reports, *, cs=(1.0, 4.0, 16.0), feature_sets=FEATURE_SETS,
        min_support=5, seed=44, max_events=None, log=print):
    labels = family_vocab(train_reports, min_support)
    if not labels:
        raise ValueError('no family meets min_support')
    train_y, train_mask = family_targets(train_reports, labels)
    valid_y, valid_mask = family_targets(valid_reports, labels)
    supervised = train_mask[:, 0] > 0
    priors = {label: float(train_y[supervised, i].mean()) for i, label in enumerate(labels)}
    train_tokens = [report_tokens(r, max_events) for r, keep in zip(train_reports, supervised) if keep]
    valid_tokens = [report_tokens(r, max_events) for r in valid_reports]
    y = train_y[supervised]
    runs = []
    for features in feature_sets:
        pick = (lambda t: t['events'] + t['evidence']) if features == 'all' else (lambda t, k=features: t[k])
        train_docs, valid_docs = [pick(t) for t in train_tokens], [pick(t) for t in valid_tokens]
        for c in cs:
            started = time.perf_counter()
            scores, n_features = fit_predict(train_docs, y, valid_docs, c, seed)
            metrics = evaluate(valid_y, scores, valid_mask, labels, priors)
            entry = {'features': features, 'C': c, 'n_features': n_features,
                     'fit_seconds': round(time.perf_counter() - started, 1), 'validation': metrics}
            runs.append(entry)
            log(json.dumps({'stage': 'classical', **{k: v for k, v in entry.items() if k != 'validation'},
                            **{k: metrics[k] for k in ('family_map', 'family_top1_accuracy',
                                                       'family_top3_accuracy', 'family_macro_f1')}}))
    best = max(runs, key=lambda r: r['validation']['family_map'] or -1)
    return {'labels': labels, 'n_train_supervised': int(supervised.sum()),
            'n_validation': len(valid_reports), 'selection_metric': 'family_map',
            'selected': {'features': best['features'], 'C': best['C']}, 'runs': runs}


def main():
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from train_winmet_expansion import load_expansion

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--C', type=float, nargs='+', default=[1.0, 4.0, 16.0])
    parser.add_argument('--features', nargs='+', choices=FEATURE_SETS, default=list(FEATURE_SETS))
    parser.add_argument('--max-events', type=int, default=None,
                        help='truncate events like the deep model (2048); default keeps all')
    parser.add_argument('--seed', type=int, default=44)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('refusing to overwrite output')
    manifest = json.loads(args.manifest.read_text())
    if manifest.get('original_split_verified') is not True:
        parser.error('verified expansion manifest required')
    started = time.perf_counter()
    reports = load_expansion(manifest.get('source'), manifest, pack=True)  # never loads withheld partitions
    load_seconds = round(time.perf_counter() - started, 1)
    print(json.dumps({'stage': 'loaded', 'seconds': load_seconds}), flush=True)
    result = run(reports['train'], reports['validation'], cs=args.C, feature_sets=args.features,
                 seed=args.seed, max_events=args.max_events, log=lambda line: print(line, flush=True))
    result.update(manifest=str(args.manifest), load_seconds=load_seconds, max_events=args.max_events,
                  total_seconds=round(time.perf_counter() - started, 1),
                  notice='Validation only. Withheld test partitions were not parsed or scored.')
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps({'stage': 'done', 'selected': result['selected'], 'output': str(args.output)}), flush=True)


if __name__ == '__main__':
    main()
