"""TLSH-routed hybrid: copy the nearest train label when a near-duplicate exists.

For each validation report, the nearest supervised train report by TLSH
distance is found. Below a distance threshold the hybrid predicts that train
report's in-vocabulary family; otherwise it uses the frozen TF-IDF + LR model
(all features, C=16). The threshold is chosen by archive-group cross-fitting:
picked on half of the validation groups and scored on the other half, then the
halves swap, so the reported top-1 is not tuned on the rows it is scored on.
Validation only; withheld test partitions are never parsed.
"""
from __future__ import annotations
import argparse
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

THRESHOLDS = (0, 10, 20, 30, 40, 50, 70, 100)


def nearest_neighbours(train, valid, labels, distance):
    """(distance, predicted label index) of the nearest labelled train report, or (inf, -1)."""
    index = {label: i for i, label in enumerate(labels)}
    reference = []
    for report in train:
        digest = (report.metadata.get('fuzzy_hashes') or {}).get('tlsh')
        family = next((index[f] for f in report.families if f in index), None) if report.family_supervised else None
        if digest and family is not None:
            reference.append((digest, family))
    out = []
    for report in valid:
        digest = (report.metadata.get('fuzzy_hashes') or {}).get('tlsh')
        if not digest or not reference:
            out.append((float('inf'), -1))
            continue
        out.append(min((distance(digest, ref), family) for ref, family in reference))
    return out


def hybrid_correct(lr_scores, neighbours, known, threshold):
    """Per-row top-1 correctness of the hybrid (NaN for rows without an in-vocabulary family)."""
    correct = np.full(len(known), np.nan)
    for row, (dist, family) in enumerate(neighbours):
        if not known[row].any():
            continue
        prediction = family if family >= 0 and dist <= threshold else int(np.argmax(lr_scores[row]))
        correct[row] = known[row, prediction]
    return correct


def cross_fitted_threshold(lr_scores, neighbours, known, groups, thresholds=THRESHOLDS):
    ordered = sorted(set(groups))
    fold = np.asarray([ordered.index(g) % 2 for g in groups])
    per_threshold = {t: hybrid_correct(lr_scores, neighbours, known, t) for t in thresholds}
    chosen, hybrid, lr_only = {}, [], []
    lr_correct = hybrid_correct(lr_scores, neighbours, known, -1)  # threshold -1 never routes
    for held in (0, 1):
        fit, test = fold != held, fold == held
        best = max(thresholds, key=lambda t: np.nanmean(per_threshold[t][fit]))
        chosen[held] = best
        hybrid.extend(v for v in per_threshold[best][test] if not np.isnan(v))
        lr_only.extend(v for v in lr_correct[test] if not np.isnan(v))
    return {'hybrid_top1': float(np.mean(hybrid)), 'lr_top1_same_rows': float(np.mean(lr_only)),
            'rows': len(hybrid), 'fold_thresholds': chosen,
            'per_threshold_full_validation': {str(t): float(np.nanmean(v)) for t, v in per_threshold.items()},
            'routed_share': {str(t): float(np.mean([d <= t for d, _ in neighbours])) for t in thresholds}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('refusing to overwrite output')
    import tlsh
    manifest = json.loads(args.manifest.read_text())
    reports = load_expansion(manifest.get('source'), manifest, pack=True)  # no withheld partitions
    train, valid = reports['train'], reports['validation']
    labels = classical.family_vocab(train, 5)
    train_y, train_mask = classical.family_targets(train, labels)
    valid_y, valid_mask = classical.family_targets(valid, labels)
    keep = train_mask[:, 0] > 0
    pick = lambda t: t['events'] + t['evidence']
    lr_scores, _ = classical.fit_predict([pick(classical.report_tokens(r)) for r, k in zip(train, keep) if k],
                                         train_y[keep], [pick(classical.report_tokens(r)) for r in valid], 16.0, 44)
    known = (valid_mask > 0) & (valid_y >= 0.5)
    neighbours = nearest_neighbours(train, valid, labels, tlsh.diff)
    groups = [r.metadata.get('archive_group_id') for r in valid]
    result = cross_fitted_threshold(lr_scores, neighbours, known, groups)
    result['notice'] = ('Validation only. hybrid_top1 uses thresholds chosen on the other half of the archive '
                        'groups; per_threshold_full_validation is optimistic and only descriptive.')
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=1), flush=True)


if __name__ == '__main__':
    main()
