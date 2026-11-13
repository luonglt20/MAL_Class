"""Multi-label behavior baseline: predict CAPE behavioral signatures from events.

Targets are the behavioral CAPE signature names kept by normalization v5
(``metadata["cape_signatures"]``); a report with at least one signature is a
supervised row where absent signatures are negatives (CAPE evaluates every
signature on every report). Features are execution events only, so signature
names never enter the input. Two encodings are compared with one fixed model
setting (no tuning on validation):

- ``unigram``: bag of event tokens (no order information);
- ``bigram``: event tokens plus adjacent event pairs (local order).

Validation only; withheld test partitions are never parsed.
"""
from __future__ import annotations
import argparse
from collections import Counter
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from train_winmet_expansion import load_expansion  # noqa: E402
from malware_hybrid.evidence import EvidenceEngine  # noqa: E402
from malware_hybrid.metrics import average_precision, multilabel_f1, multilabel_metrics  # noqa: E402

_spec = importlib.util.spec_from_file_location('classical_family_baseline', ROOT / 'classical_family_baseline.py')
classical = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(classical)

ENCODINGS = ('unigram', 'bigram')


def signature_vocab(reports, min_support):
    counts = Counter()
    for report in reports:
        counts.update(set(report.metadata.get('cape_signatures') or []))
    return sorted(name for name, count in counts.items() if count >= min_support)


def signature_targets(reports, labels):
    index = {label: i for i, label in enumerate(labels)}
    targets = np.zeros((len(reports), len(labels)))
    mask = np.zeros_like(targets)
    for row, report in enumerate(reports):
        names = report.metadata.get('cape_signatures') or []
        if names:
            mask[row] = 1.0
            for name in names:
                if name in index:
                    targets[row, index[name]] = 1.0
    return targets, mask


def event_tokens(report, encoding):
    tokens = classical.report_tokens(report)['events']
    return [t for t in tokens if t.startswith('e:')] if encoding == 'unigram' else tokens


def evaluate(targets, scores, mask, labels):
    ranking = multilabel_metrics(targets, scores, mask, labels, prefix='signature')
    f1 = multilabel_f1(targets, scores, mask)
    return {'signature_map': ranking['signature_map'], 'micro_f1': f1['micro_f1'], 'macro_f1': f1['macro_f1'],
            'per_label_ap': {label: ranking['per_signature'][label]['average_precision'] for label in labels}}


def bootstrap_map_difference(targets, a, b, mask, rounds=200, seed=44):
    """Paired bootstrap over reports of mAP(b) - mAP(a)."""
    rng = np.random.default_rng(seed)
    rows = np.flatnonzero(mask.any(axis=1))
    diffs = []
    for _ in range(rounds):
        pick = rng.choice(rows, size=len(rows), replace=True)
        values = []
        for column in range(targets.shape[1]):
            truth = targets[pick, column] > 0
            ap_a, ap_b = average_precision(truth, a[pick, column]), average_precision(truth, b[pick, column])
            if ap_a is not None and ap_b is not None:
                values.append(ap_b - ap_a)
        diffs.append(float(np.mean(values)))
    low, high = np.percentile(diffs, [2.5, 97.5])
    return {'mean': float(np.mean(diffs)), 'ci95': [float(low), float(high)], 'rounds': rounds}


def rule_agreement(reports, technique, signature):
    engine, table = EvidenceEngine(), Counter()
    for report in reports:
        names = set(report.metadata.get('cape_signatures') or [])
        if not names:
            continue
        status = engine.evaluate(report.events)[technique].status if report.events else 'absent'
        table[f'{status}|{"signature" if signature in names else "no_signature"}'] += 1
    return dict(table)


def run(train, valid, *, min_support=200, c=4.0, seed=44, log=print):
    labels = signature_vocab(train, min_support)
    if not labels:
        raise ValueError('no signature meets min_support')
    train_y, train_mask = signature_targets(train, labels)
    valid_y, valid_mask = signature_targets(valid, labels)
    keep = train_mask[:, 0] > 0
    results, scores = {}, {}
    for encoding in ENCODINGS:
        started = time.perf_counter()
        train_docs = [event_tokens(r, encoding) for r, k in zip(train, keep) if k]
        valid_docs = [event_tokens(r, encoding) for r in valid]
        scores[encoding], n_features = classical.fit_predict(train_docs, train_y[keep], valid_docs, c, seed)
        results[encoding] = {'n_features': n_features, 'fit_seconds': round(time.perf_counter() - started, 1),
                             'validation': evaluate(valid_y, scores[encoding], valid_mask, labels)}
        log(json.dumps({'stage': 'behavior', 'encoding': encoding, 'n_features': n_features,
                        **{k: results[encoding]['validation'][k] for k in ('signature_map', 'micro_f1', 'macro_f1')}}))
    a, b = results['unigram']['validation']['per_label_ap'], results['bigram']['validation']['per_label_ap']
    compared = [label for label in labels if a[label] is not None and b[label] is not None]
    return {
        'labels': labels, 'min_support': min_support, 'C': c,
        'n_train_supervised': int(keep.sum()), 'n_validation_supervised': int(valid_mask[:, 0].sum()),
        'encodings': results,
        'bigram_minus_unigram': {
            'labels_better': sum(b[l] > a[l] for l in compared), 'labels_worse': sum(b[l] < a[l] for l in compared),
            'paired_bootstrap_map': bootstrap_map_difference(valid_y, scores['unigram'], scores['bigram'], valid_mask),
            'largest_gains': sorted(((b[l] - a[l], l) for l in compared), reverse=True)[:10],
        },
        'rule_vs_signature_validation': {'T1547.001|persistence_autorun': rule_agreement(valid, 'T1547.001',
                                                                                         'persistence_autorun')},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--min-support', type=int, default=200)
    parser.add_argument('--C', type=float, default=4.0)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('refusing to overwrite output')
    manifest = json.loads(args.manifest.read_text())
    reports = load_expansion(manifest.get('source'), manifest, pack=True)  # no withheld partitions
    result = run(reports['train'], reports['validation'], min_support=args.min_support, c=args.C,
                 log=lambda line: print(line, flush=True))
    result['notice'] = ('CAPE signatures are heuristic weak labels, not ground truth. Validation only; '
                        'one fixed model setting, no tuning on validation.')
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: result[k] for k in ('bigram_minus_unigram', 'rule_vs_signature_validation')}), flush=True)


if __name__ == '__main__':
    main()
