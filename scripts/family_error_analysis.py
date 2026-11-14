"""Where does the selected TF-IDF + LR family model fail on validation?

Refits the frozen classical configuration (all features, C=16) on train and
breaks validation top-1 errors down by train support of the true family, the
most frequent confusion pairs, whether the report's imphash also occurs in
train (near-duplicate builds), trace length and multi-label rows. It also
checks a TLSH nearest-neighbour readout when the optional ``tlsh`` package is
installed. Validation only; withheld test partitions are never parsed.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from train_winmet_expansion import load_expansion  # noqa: E402

_spec = importlib.util.spec_from_file_location('classical_family_baseline', ROOT / 'classical_family_baseline.py')
classical = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(classical)

SUPPORT_BUCKETS = ((5, 19), (20, 49), (50, 199), (200, 10 ** 9))
LENGTH_BUCKETS = ((0, 99), (100, 499), (500, 1999), (2000, 10 ** 9))


def _bucket(value, buckets):
    for low, high in buckets:
        if low <= value <= high:
            return f'{low}-{high if high < 10 ** 9 else "max"}'
    return 'other'


def _rate(rows):
    return {'rows': len(rows), 'top1': round(float(np.mean(rows)), 4) if rows else None}


def analyse(valid, train, labels, scores, targets, mask):
    known = (mask > 0) & (targets >= 0.5)
    rows = np.flatnonzero(known.any(axis=1))
    top = np.argmax(scores, axis=1)
    support = Counter(f for r in train if r.family_supervised for f in r.families)
    train_imphash = {(r.metadata.get('fuzzy_hashes') or {}).get('imphash') for r in train} - {None, ''}
    by_support, by_length, by_imphash, by_multi = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
    confusions, per_family = Counter(), defaultdict(list)
    for i in rows:
        report = valid[i]
        correct = bool(known[i, top[i]])
        true = [labels[j] for j in np.flatnonzero(known[i])]
        main = max(true, key=lambda f: support[f])
        by_support[_bucket(support[main], SUPPORT_BUCKETS)].append(correct)
        by_length[_bucket(len(report.events), LENGTH_BUCKETS)].append(correct)
        imphash = (report.metadata.get('fuzzy_hashes') or {}).get('imphash')
        by_imphash['imphash_seen_in_train' if imphash in train_imphash else
                   'imphash_unseen_or_missing'].append(correct)
        by_multi['multi_label' if len(report.families) > 1 else 'single_label'].append(correct)
        per_family[main].append(correct)
        if not correct:
            confusions[(main, labels[top[i]])] += 1
    worst = sorted(((len(v) - sum(v), f, len(v)) for f, v in per_family.items()), reverse=True)[:15]
    return {
        'validation_rows': int(len(rows)), 'top1': round(float(np.mean([bool(known[i, top[i]]) for i in rows])), 4),
        'errors': int(sum(not known[i, top[i]] for i in rows)),
        'by_true_family_train_support': {k: _rate(v) for k, v in sorted(by_support.items())},
        'by_trace_length': {k: _rate(v) for k, v in sorted(by_length.items())},
        'by_imphash_overlap': {k: _rate(v) for k, v in by_imphash.items()},
        'by_label_count': {k: _rate(v) for k, v in by_multi.items()},
        'top_confusions_true_to_predicted': [[a, b, n] for (a, b), n in confusions.most_common(20)],
        'families_with_most_errors': [{'family': f, 'errors': e, 'rows': n, 'train_support': support[f]}
                                      for e, f, n in worst],
    }


def tlsh_nearest(train, valid, labels, targets, mask):
    try:
        import tlsh
    except ImportError:
        return {'available': False, 'reason': 'tlsh package not installed'}
    index = {label: i for i, label in enumerate(labels)}
    reference = [((r.metadata.get('fuzzy_hashes') or {}).get('tlsh'), r.families) for r in train if r.family_supervised]
    reference = [(h, fams) for h, fams in reference if h and any(f in index for f in fams)]
    known = (mask > 0) & (targets >= 0.5)
    hits, coverage = [], 0
    for i, report in enumerate(valid):
        if not known[i].any():
            continue
        digest = (report.metadata.get('fuzzy_hashes') or {}).get('tlsh')
        if not digest:
            continue
        coverage += 1
        distance, families = min((tlsh.diff(digest, h), fams) for h, fams in reference)
        predicted = next(f for f in families if f in index)
        hits.append((distance, bool(known[i, index[predicted]])))
    result = {'available': True, 'rows_with_tlsh': coverage}
    for limit in (30, 50, 100, 10 ** 9):
        chosen = [ok for d, ok in hits if d <= limit]
        result[f'distance<= {limit}'] = {'coverage': len(chosen), 'top1': round(float(np.mean(chosen)), 4) if chosen else None}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('refusing to overwrite output')
    manifest = json.loads(args.manifest.read_text())
    reports = load_expansion(manifest.get('source'), manifest, pack=True)  # no withheld partitions
    train, valid = reports['train'], reports['validation']
    labels = classical.family_vocab(train, 5)
    train_y, train_mask = classical.family_targets(train, labels)
    valid_y, valid_mask = classical.family_targets(valid, labels)
    keep = train_mask[:, 0] > 0
    pick = lambda t: t['events'] + t['evidence']
    train_docs = [pick(classical.report_tokens(r)) for r, k in zip(train, keep) if k]
    valid_docs = [pick(classical.report_tokens(r)) for r in valid]
    scores, _ = classical.fit_predict(train_docs, train_y[keep], valid_docs, 16.0, 44)
    result = analyse(valid, train, labels, scores, valid_y, valid_mask)
    result['tlsh_nearest_neighbour'] = tlsh_nearest(train, valid, labels, valid_y, valid_mask)
    result['notice'] = 'Validation only; AVClass labels are weak; imphash/TLSH come from WinMET metadata.'
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=1)[:6000], flush=True)


if __name__ == '__main__':
    main()
