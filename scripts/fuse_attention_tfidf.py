"""Late fusion of the Hybrid Attention checkpoint with the TF-IDF + LR baseline.

``select`` scores validation only: it picks the fusion weight by validation family
mAP, reports a group cross-fitted estimate of that choice, and freezes the
selection to JSON. ``test`` scores the frozen selection once on withheld test
partitions; it fits nothing on test and refuses to run twice (lock file).
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from train_winmet_expansion import load_expansion  # noqa: E402
from malware_hybrid.dataset import HybridDataset, collate_samples  # noqa: E402
from malware_hybrid.training import _move, load_checkpoint  # noqa: E402

_spec = importlib.util.spec_from_file_location('classical_family_baseline', ROOT / 'classical_family_baseline.py')
classical = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(classical)

WEIGHTS = tuple(round(w / 10, 1) for w in range(11))  # weight of the attention scores


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def attention_scores(model, tensorizer, reports, labels, device, batch_size=4, release_every=25):
    """Sigmoid family scores with columns reordered to ``labels``."""
    vocab = tensorizer.family_vocab.token_to_id
    missing = [label for label in labels if label not in vocab]
    if missing:
        raise ValueError(f'labels missing from checkpoint vocabulary: {missing[:5]}')
    columns = [vocab[label] for label in labels]
    loader = DataLoader(HybridDataset(reports, tensorizer), batch_size=batch_size,
                        shuffle=False, collate_fn=collate_samples)
    out = []
    model.eval()
    with torch.inference_mode():
        for index, batch in enumerate(loader, start=1):
            output = model(_move(batch, device), return_attention=False)
            out.append(torch.sigmoid(output['family_logits']).float().cpu().numpy()[:, columns])
            del batch, output
            if device.type == 'mps' and index % release_every == 0:
                torch.mps.empty_cache()
    return np.concatenate(out)


def tfidf_scores(train_reports, eval_sets, labels, features, c, seed):
    """Fit the frozen TF-IDF config on train once; score each evaluation set."""
    pick = (lambda t: t['events'] + t['evidence']) if features == 'all' else (lambda t: t[features])
    train_y, train_mask = classical.family_targets(train_reports, labels)
    keep = train_mask[:, 0] > 0
    train_docs = [pick(classical.report_tokens(r)) for r, k in zip(train_reports, keep) if k]
    docs = [[pick(classical.report_tokens(r)) for r in reports] for reports in eval_sets]
    flat = [doc for group in docs for doc in group]
    scores, _ = classical.fit_predict(train_docs, train_y[keep], flat, c, seed)
    result, start = [], 0
    for group in docs:
        result.append(scores[start:start + len(group)])
        start += len(group)
    return result, {label: float(train_y[keep, i].mean()) for i, label in enumerate(labels)}


def fuse(att, tfidf, weight):
    return weight * att + (1.0 - weight) * tfidf


def sweep(att, tfidf, targets, mask, labels, priors, weights=WEIGHTS):
    return {str(w): classical.evaluate(targets, fuse(att, tfidf, w), mask, labels, priors) for w in weights}


def best_weight(att, tfidf, targets, mask, labels, priors, weights=WEIGHTS):
    results = sweep(att, tfidf, targets, mask, labels, priors, weights)
    return max(weights, key=lambda w: results[str(w)]['family_map'] or -1.0), results


def cross_fitted(att, tfidf, targets, mask, groups, labels, priors):
    """Two group folds: choose the weight on one fold, score it on the other."""
    ordered = sorted(set(groups))
    fold_of = {group: i % 2 for i, group in enumerate(ordered)}
    folds = np.asarray([fold_of[g] for g in groups])
    fused = np.zeros_like(att)
    chosen = {}
    for fold in (0, 1):
        fit, held = folds != fold, folds == fold
        w, _ = best_weight(att[fit], tfidf[fit], targets[fit], mask[fit], labels, priors)
        chosen[fold] = w
        fused[held] = fuse(att[held], tfidf[held], w)
    return classical.evaluate(targets, fused, mask, labels, priors), chosen, len(ordered)


def _model(checkpoint, device):
    model, tensorizer = load_checkpoint(checkpoint, device)
    return model.to(device), tensorizer


def select(args):
    if args.selection.exists():
        raise SystemExit('refusing to overwrite selection')
    manifest = json.loads(args.manifest.read_text())
    classical_report = json.loads(args.classical_report.read_text())
    config = classical_report['selected']
    labels = classical_report['labels']
    started = time.perf_counter()
    reports = load_expansion(manifest.get('source'), manifest, pack=True)
    train, valid = reports['train'], reports['validation']
    if classical.family_vocab(train, 5) != labels:
        raise SystemExit('label set differs from the classical report')
    device = torch.device(args.device)
    model, tensorizer = _model(args.checkpoint, args.device)
    print(json.dumps({'stage': 'loaded', 'seconds': round(time.perf_counter() - started)}), flush=True)
    att = attention_scores(model, tensorizer, valid, labels, device)
    print(json.dumps({'stage': 'attention_scored'}), flush=True)
    (tf,), priors = tfidf_scores(train, [valid], labels, config['features'], config['C'], args.seed)
    targets, mask = classical.family_targets(valid, labels)
    groups = [r.metadata.get('archive_group_id') for r in valid]
    weight, results = best_weight(att, tf, targets, mask, labels, priors)
    cross, chosen, n_groups = cross_fitted(att, tf, targets, mask, groups, labels, priors)
    selection = {
        'status': 'validation_only_frozen_selection',
        'checkpoint': str(args.checkpoint), 'checkpoint_sha256': sha256(args.checkpoint),
        'manifest_sha256': sha256(args.manifest), 'classical_report_sha256': sha256(args.classical_report),
        'tfidf_config': config, 'seed': args.seed, 'labels': labels,
        'selection_metric': 'family_map', 'weights_tried': list(WEIGHTS),
        'selected_attention_weight': weight,
        'validation_by_weight': results,
        'group_cross_fitted': {'validation': cross, 'fold_weights': chosen, 'archive_groups': n_groups},
        'seconds': round(time.perf_counter() - started),
        'test_metrics': None,
    }
    with args.selection.open('x') as handle:
        json.dump(selection, handle, indent=2)
    pick = lambda w: {k: results[str(w)][k] for k in ('family_top1_accuracy', 'family_top3_accuracy', 'family_map')}
    print(json.dumps({'stage': 'done', 'weight': weight, 'attention_only': pick(1.0), 'tfidf_only': pick(0.0),
                      'fused': pick(weight), 'cross_fitted_top1': cross['family_top1_accuracy'],
                      'fold_weights': chosen}), flush=True)


def test(args):
    selection = json.loads(args.selection.read_text())
    lock = args.selection.with_suffix('.test_evaluated.lock')
    if args.output.exists() or lock.exists():
        raise SystemExit('test already evaluated for this selection; refusing to rescore')
    if selection.get('test_metrics') is not None or selection.get('status') != 'validation_only_frozen_selection':
        raise SystemExit('a validation-only frozen selection is required')
    if sha256(args.checkpoint) != selection['checkpoint_sha256'] or sha256(args.manifest) != selection['manifest_sha256']:
        raise SystemExit('checkpoint or manifest differs from the frozen selection')
    lock.write_text(json.dumps({'output': str(args.output), 'selection_sha256': sha256(args.selection)}))
    manifest = json.loads(args.manifest.read_text())
    labels, config, weight = selection['labels'], selection['tfidf_config'], selection['selected_attention_weight']
    device = torch.device(args.device)
    model, tensorizer = _model(args.checkpoint, args.device)
    output = {'status': 'one_time_independent_test', 'selection_sha256': sha256(args.selection),
              'selected_attention_weight': weight, 'tfidf_config': config,
              'label_scope': f'{len(labels)} families with train support >= 5', 'partitions': {}}
    for name in args.partitions:
        reports = load_expansion(manifest.get('source'), manifest, include_test=name, pack=True)
        test_reports = reports[name]
        att = attention_scores(model, tensorizer, test_reports, labels, device)
        (tf,), priors = tfidf_scores(reports['train'], [test_reports], labels,
                                     config['features'], config['C'], selection['seed'])
        targets, mask = classical.family_targets(test_reports, labels)
        output['partitions'][name] = {
            'rows': len(test_reports),
            'attention_only': classical.evaluate(targets, att, mask, labels, priors),
            'tfidf_only': classical.evaluate(targets, tf, mask, labels, priors),
            'fused_frozen_weight': classical.evaluate(targets, fuse(att, tf, weight), mask, labels, priors),
        }
        print(json.dumps({'stage': 'partition', 'name': name, **{
            k: {m: v[m] for m in ('family_top1_accuracy', 'family_top3_accuracy', 'family_map', 'family_evaluated_samples')}
            for k, v in output['partitions'][name].items() if isinstance(v, dict)}}), flush=True)
        del reports, test_reports
    output['notice'] = ('Scored once with the model, TF-IDF config and fusion weight frozen on validation. '
                        'Top-1/top-3 denominators are test rows with an in-vocabulary family.')
    with args.output.open('x') as handle:
        json.dump(output, handle, indent=2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('select', 'test'):
        p = sub.add_parser(name)
        p.add_argument('manifest', type=Path)
        p.add_argument('checkpoint', type=Path)
        p.add_argument('--selection', type=Path, required=True)
        p.add_argument('--device', default='mps')
    sub.choices['select'].add_argument('--classical-report', type=Path, required=True)
    sub.choices['select'].add_argument('--seed', type=int, default=44)
    sub.choices['test'].add_argument('--output', type=Path, required=True)
    sub.choices['test'].add_argument('--partitions', nargs='+',
                                     default=['withheld_internal_test', 'withheld_test_v2'])
    args = parser.parse_args()
    torch.set_num_threads(4)
    (select if args.command == 'select' else test)(args)


if __name__ == '__main__':
    main()
