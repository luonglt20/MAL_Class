"""Report family top-1 alongside top-1 over train-derived groups of confusable families.

Groups are built from train only: out-of-fold predictions of the frozen TF-IDF +
LR configuration (archive-group k-fold inside train) give a confusion matrix, and
two families whose mutual confusion rate r(a->b) + r(b->a) reaches a threshold
fixed in advance are merged (union-find). Validation is then scored at the
family level (unchanged task) and at the group level (a coarser, separately
reported task). Validation never influences the groups. Withheld test
partitions are never parsed.
"""
from __future__ import annotations
import argparse
from collections import Counter
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from train_winmet_expansion import load_expansion  # noqa: E402

_spec = importlib.util.spec_from_file_location('classical_family_baseline', ROOT / 'classical_family_baseline.py')
classical = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(classical)

MERGE_THRESHOLD = 0.30  # fixed before looking at any grouped result


def merge_groups(confusion, labels, threshold=MERGE_THRESHOLD):
    """Union-find over families whose mutual out-of-fold confusion reaches ``threshold``."""
    totals = confusion.sum(axis=1)
    parent = list(range(len(labels)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    merged = []
    for a in range(len(labels)):
        for b in range(a + 1, len(labels)):
            rate = (confusion[a, b] / totals[a] if totals[a] else 0.0) + (confusion[b, a] / totals[b] if totals[b] else 0.0)
            if rate >= threshold:
                parent[find(a)] = find(b)
                merged.append([labels[a], labels[b], round(float(rate), 3)])
    return [find(i) for i in range(len(labels))], merged


def out_of_fold_confusion(docs, y, groups, folds, c, seed):
    """Confusion counts (true main family -> predicted family) from archive-group k-fold inside train."""
    n_labels = y.shape[1]
    confusion = np.zeros((n_labels, n_labels))
    for fit, held in GroupKFold(n_splits=folds).split(docs, groups=groups):
        present = y[fit].sum(axis=0) > 0
        scores, _ = classical.fit_predict([docs[i] for i in fit], y[fit][:, present], [docs[i] for i in held], c, seed)
        full = np.full((len(held), n_labels), -1.0)
        full[:, present] = scores
        for row, index in enumerate(held):
            true = np.flatnonzero(y[index] > 0)
            if len(true):
                confusion[true[0], int(np.argmax(full[row]))] += 1
    return confusion


def grouped_top1(scores, targets, mask, group_of):
    known = (mask > 0) & (targets >= 0.5)
    rows = np.flatnonzero(known.any(axis=1))
    top = np.argmax(scores, axis=1)
    fine = [bool(known[i, top[i]]) for i in rows]
    coarse = [group_of[top[i]] in {group_of[j] for j in np.flatnonzero(known[i])} for i in rows]
    return float(np.mean(fine)), float(np.mean(coarse)), len(rows)


def run(train, valid, *, folds=5, c=16.0, seed=44, threshold=MERGE_THRESHOLD, log=print):
    labels = classical.family_vocab(train, 5)
    train_y, train_mask = classical.family_targets(train, labels)
    valid_y, valid_mask = classical.family_targets(valid, labels)
    keep = np.flatnonzero(train_mask[:, 0] > 0)
    pick = lambda t: t['events'] + t['evidence']
    docs = [pick(classical.report_tokens(train[i])) for i in keep]
    groups = [train[i].metadata.get('archive_group_id') for i in keep]
    confusion = out_of_fold_confusion(docs, train_y[keep], groups, folds, c, seed)
    log(json.dumps({'stage': 'out_of_fold_done'}))
    group_of, merged = merge_groups(confusion, labels, threshold)
    scores, _ = classical.fit_predict(docs, train_y[keep], [pick(classical.report_tokens(r)) for r in valid], c, seed)
    fine, coarse, rows = grouped_top1(scores, valid_y, valid_mask, group_of)
    members = Counter(group_of)
    named = {}
    for i, g in enumerate(group_of):
        if members[g] > 1:
            named.setdefault(g, []).append(labels[i])
    return {'families': len(labels), 'groups': len(members), 'merge_threshold': threshold, 'folds': folds,
            'validation_rows': rows, 'family_top1': fine, 'group_top1': coarse,
            'merged_groups': sorted(named.values(), key=len, reverse=True), 'merged_pairs': merged}


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
    result['notice'] = ('Groups come from train-only out-of-fold confusion with a threshold fixed in advance. '
                        'group_top1 is a coarser task and must be reported next to family_top1, never instead.')
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=1), flush=True)


if __name__ == '__main__':
    main()
