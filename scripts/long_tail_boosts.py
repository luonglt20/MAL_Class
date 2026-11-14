"""Long-tail fixes for family top-1, each compared with the TF-IDF + LR baseline.

Variants (validation only, same partitions, labels and metrics as the classical
baseline): class-balanced logistic regression; post-hoc logit adjustment that
divides scores by train family priors to the power tau; and either of them
combined with TLSH routing (copy the nearest train label at TLSH distance <= 30).
The variant is chosen by archive-group cross-fitting, and the per-row result of
that cross-fitted choice is compared with the baseline using an exact McNemar
test. Withheld test partitions are never parsed.
"""
from __future__ import annotations
import argparse
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
from scipy.stats import binomtest
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from train_winmet_expansion import load_expansion  # noqa: E402


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


classical = _load('classical_family_baseline')
hybrid = _load('tlsh_routed_hybrid')
TAUS = (0.25, 0.5, 1.0)
TLSH_THRESHOLD = 30


def logit_adjust(scores, priors, tau):
    return scores / np.power(np.clip(priors, 1e-6, None), tau)[None, :]


def row_correct(scores, known, neighbours=None, threshold=None):
    if neighbours is not None:
        return hybrid.hybrid_correct(scores, neighbours, known, threshold)
    out = np.full(len(known), np.nan)
    rows = known.any(axis=1)
    out[rows] = known[rows, np.argmax(scores[rows], axis=1)]
    return out


def mcnemar(baseline, candidate):
    both = ~np.isnan(baseline) & ~np.isnan(candidate)
    gained = int(((candidate == 1) & (baseline == 0) & both).sum())
    lost = int(((candidate == 0) & (baseline == 1) & both).sum())
    p = binomtest(gained, gained + lost, 0.5).pvalue if gained + lost else 1.0
    return {'gained': gained, 'lost': lost, 'p_value_exact': float(p)}


def cross_fitted(correct_by_variant, groups):
    ordered = sorted(set(groups))
    fold = np.asarray([ordered.index(g) % 2 for g in groups])
    combined = np.full(len(groups), np.nan)
    chosen = {}
    for held in (0, 1):
        fit, test = fold != held, fold == held
        best = max(correct_by_variant, key=lambda n: np.nanmean(correct_by_variant[n][fit]))
        chosen[held] = best
        combined[test] = correct_by_variant[best][test]
    return combined, chosen


def run(train, valid, *, c=16.0, seed=44, distance=None, log=print):
    labels = classical.family_vocab(train, 5)
    train_y, train_mask = classical.family_targets(train, labels)
    valid_y, valid_mask = classical.family_targets(valid, labels)
    keep = train_mask[:, 0] > 0
    priors = train_y[keep].mean(axis=0)
    known = (valid_mask > 0) & (valid_y >= 0.5)
    pick = lambda t: t['events'] + t['evidence']
    x_train, x_valid = classical.TfidfVectorizer(
        analyzer=lambda d: d, lowercase=False, min_df=2, sublinear_tf=True, max_features=300_000,
        dtype=np.float32), None
    vectorizer = x_train
    x_train = vectorizer.fit_transform([pick(classical.report_tokens(r)) for r, k in zip(train, keep) if k])
    x_valid = vectorizer.transform([pick(classical.report_tokens(r)) for r in valid])
    scores = {}
    for name, weight in (('lr', None), ('lr_balanced', 'balanced')):
        model = OneVsRestClassifier(LogisticRegression(C=c, solver='liblinear', max_iter=2000,
                                                       random_state=seed, class_weight=weight))
        scores[name] = model.fit(x_train, train_y[keep]).predict_proba(x_valid)
        log(json.dumps({'fitted': name}))
    for tau in TAUS:
        scores[f'lr_adjust_{tau:g}'] = logit_adjust(scores['lr'], priors, tau)
    neighbours = hybrid.nearest_neighbours(train, valid, labels, distance) if distance else None
    correct = {name: row_correct(s, known) for name, s in scores.items()}
    if neighbours is not None:
        for name, s in list(scores.items()):
            correct[f'{name}+tlsh{TLSH_THRESHOLD}'] = row_correct(s, known, neighbours, TLSH_THRESHOLD)
    baseline = correct['lr']
    groups = [r.metadata.get('archive_group_id') for r in valid]
    combined, chosen = cross_fitted(correct, groups)
    variants = {name: {'top1': float(np.nanmean(v)), **mcnemar(baseline, v)} for name, v in correct.items()}
    return {
        'validation_rows': int(known.any(axis=1).sum()),
        'variants_full_validation': variants,
        'cross_fitted': {'top1': float(np.nanmean(combined)), 'fold_choices': chosen,
                         'vs_baseline': mcnemar(baseline, combined)},
        'baseline_top1': float(np.nanmean(baseline)),
    }


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
    result = run(reports['train'], reports['validation'], distance=tlsh.diff,
                 log=lambda line: print(line, flush=True))
    result['notice'] = ('Validation only. variants_full_validation are optimistic when compared with each other; '
                        'cross_fitted chooses the variant on the other half of the archive groups.')
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=1), flush=True)


if __name__ == '__main__':
    main()
