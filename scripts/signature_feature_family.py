"""Do behavioral CAPE signatures add family information beyond events and static evidence?

On the v5 data, fits the frozen TF-IDF + LR configuration (C=16) with and without
the behavioral CAPE signature names (``metadata["cape_signatures"]``; family,
YARA and IDS detections were already excluded by normalization) as extra tokens,
and compares validation top-1 per report with an exact McNemar test. A
signature-only model is reported too. Withheld test partitions are never parsed.
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


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


classical = _load('classical_family_baseline')
long_tail = _load('long_tail_boosts')
FEATURES = ('base', 'base+signatures', 'signatures')


def tokens(report, features):
    base = classical.report_tokens(report)
    signatures = [f'g:{name}' for name in report.metadata.get('cape_signatures') or []]
    if features == 'signatures':
        return signatures
    return base['events'] + base['evidence'] + (signatures if features == 'base+signatures' else [])


def run(train, valid, *, c=16.0, seed=44, log=print):
    labels = classical.family_vocab(train, 5)
    train_y, train_mask = classical.family_targets(train, labels)
    valid_y, valid_mask = classical.family_targets(valid, labels)
    keep = train_mask[:, 0] > 0
    priors = {label: float(train_y[keep, i].mean()) for i, label in enumerate(labels)}
    known = (valid_mask > 0) & (valid_y >= 0.5)
    results, correct = {}, {}
    for features in FEATURES:
        scores, n_features = classical.fit_predict(
            [tokens(r, features) for r, k in zip(train, keep) if k], train_y[keep],
            [tokens(r, features) for r in valid], c, seed)
        metrics = classical.evaluate(valid_y, scores, valid_mask, labels, priors)
        correct[features] = long_tail.row_correct(scores, known)
        results[features] = {'n_features': n_features, **{k: metrics[k] for k in (
            'family_top1_accuracy', 'family_top3_accuracy', 'family_map', 'family_macro_f1',
            'family_evaluated_samples')}}
        log(json.dumps({'features': features, **results[features]}))
    return {'labels': len(labels), 'features': results,
            'base+signatures_vs_base': long_tail.mcnemar(correct['base'], correct['base+signatures'])}


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
    result['notice'] = 'Validation only, v5 data; one fixed C, no tuning on validation.'
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=1), flush=True)


if __name__ == '__main__':
    main()
