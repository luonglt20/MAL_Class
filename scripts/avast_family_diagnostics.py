"""Per-family validation diagnostics for the frozen Avast-CTU models.

Validation only: the withheld test partition is never loaded. Reports, for
every family, the number of validation rows, the event-count distribution of
the prepared reports (to check the shared 2,048-event cap) and top-1 accuracy
of the frozen TF-IDF configuration and of the attention checkpoint.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import classical_family_baseline as classical  # noqa: E402
from fuse_attention_tfidf import _model, attention_scores, tfidf_scores  # noqa: E402
from train_winmet_expansion import load_expansion  # noqa: E402


def per_family_top1(scores, targets, mask, labels):
    known = (mask > 0) & (targets >= 0.5)
    rows = known.any(axis=1)
    top = scores.argmax(axis=1)
    out = {}
    for i, label in enumerate(labels):
        members = rows & known[:, i]
        if members.any():
            out[label] = {'rows': int(members.sum()),
                          'top1': float(known[members, top[members]].mean())}
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='mps')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('refusing to overwrite output')
    manifest = json.loads(args.manifest.read_text())
    selection = json.loads(args.selection.read_text())
    labels = selection['labels']
    reports = load_expansion(manifest.get('source'), manifest, pack=True)  # never loads withheld partitions
    train, valid = reports['train'], reports['validation']
    family_of = []
    lengths = {}
    for report in valid:
        family = report.families[0] if report.families else None
        family_of.append(family)
        lengths.setdefault(family, []).append(len(report.events))
    event_stats = {f: {'median': float(np.median(v)), 'p90': float(np.percentile(v, 90)), 'max': int(max(v)),
                       'share_at_cap': float(np.mean(np.asarray(v) >= 2048))} for f, v in lengths.items()}
    print(json.dumps({'stage': 'events', 'stats': event_stats}), flush=True)
    targets, mask = classical.family_targets(valid, labels)
    cfg = selection['tfidf_config']
    (tfidf,), _ = tfidf_scores(train, [valid], labels, cfg['features'], cfg['C'], selection['seed'])
    print(json.dumps({'stage': 'tfidf_scored'}), flush=True)
    device = torch.device(args.device)
    model, tensorizer = _model(args.checkpoint, args.device)
    att = attention_scores(model, tensorizer, valid, labels, device)
    result = {
        'status': 'validation_only_diagnostics', 'manifest': str(args.manifest), 'checkpoint': str(args.checkpoint),
        'tfidf_config': cfg, 'labels': labels, 'event_counts': event_stats,
        'tfidf_per_family': per_family_top1(tfidf, targets, mask, labels),
        'attention_per_family': per_family_top1(att, targets, mask, labels),
        'notice': 'Validation only; the withheld test partition was not loaded.',
    }
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps({'stage': 'done', 'output': str(args.output)}), flush=True)


if __name__ == '__main__':
    main()
