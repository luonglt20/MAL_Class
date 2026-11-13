"""Train one expanded Hybrid Attention candidate with frozen archive partitions."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import torch

from malware_hybrid.graph import ExecutionGraphBuilder
from malware_hybrid.packed_reports import PackedReports
from malware_hybrid.serialization import report_from_dict
from malware_hybrid.training import train_model


def _withheld(name):
    return name.startswith('withheld')


def load_expansion(source, manifest, *, include_test=False, pack=False):
    """Load frozen partitions; withheld partitions stay unparsed unless requested.

    ``include_test`` is False, True (the original ``withheld_internal_test``) or a
    withheld partition name. ``pack`` stores train and validation reports
    compressed (see ``PackedReports``) so the raw corpus is never held in RAM;
    calibration stays a plain list. Manifests with ``sources`` (list of {path, sha256})
    combine several prepared JSONL files; otherwise ``source`` and
    ``source_sha256`` describe one file.
    """
    wanted = 'withheld_internal_test' if include_test is True else (include_test or None)
    if wanted is not None and wanted not in manifest['partitions']:
        raise ValueError('requested withheld partition is not in the manifest')
    groups = {}
    for name, part in manifest['partitions'].items():
        for group in part['archive_groups']:
            if group in groups:
                raise ValueError('archive group overlaps partitions')
            groups[group] = name
    retained = {
        name: PackedReports() if pack and name in ('train', 'validation') else []
        for name in manifest['partitions'] if not _withheld(name)
    }
    if wanted is not None:
        # Only the one-time frozen test evaluator may request a withheld partition.
        retained[wanted] = PackedReports() if pack else []
    identities = {name: [] for name in manifest['partitions']}
    sources = manifest.get('sources') or [{'path': source, 'sha256': manifest['source_sha256']}]
    seen = set()
    # Rows duplicated across prepared files (same SHA-256) are excluded by the
    # manifest builder; their lines still count toward the source checksum.
    excluded = {(item['source'], item['sample_hash']) for item in manifest.get('excluded_rows', [])}
    builder = ExecutionGraphBuilder()
    loaded = 0
    for entry in sources:
        digest = hashlib.sha256()
        with Path(entry['path']).open('rb') as handle:
            for line in handle:
                digest.update(line)
                payload = json.loads(line)
                identity = payload['sample_hash']
                if (str(entry['path']), identity) in excluded:
                    continue
                if identity in seen:
                    raise ValueError('duplicate sample identity')
                seen.add(identity)
                group = payload.get('metadata', {}).get('archive_group_id')
                if group not in groups:
                    raise ValueError('unassigned archive group')
                name = groups[group]
                identities[name].append(identity)
                loaded += 1
                if name not in retained:
                    # Do not construct withheld event/graph objects or predictions.
                    continue
                report = report_from_dict(payload)
                canonical = builder.build(report.events)
                if report.graph is not None and report.graph != canonical:
                    raise ValueError('stored graph differs from event reconstruction')
                report.graph = None  # Rebuild on demand; avoids retaining duplicate graph objects.
                retained[name].append(report)
                if loaded % 250 == 0:
                    print(json.dumps({'loaded_rows': loaded}), flush=True)
        if digest.hexdigest() != entry['sha256']:
            raise ValueError('prepared source checksum mismatch')
    for name, hashes in identities.items():
        actual = hashlib.sha256('\n'.join(sorted(hashes)).encode()).hexdigest()
        if len(hashes) != manifest['partitions'][name]['rows'] or actual != manifest['partitions'][name]['sample_ids_sha256']:
            raise ValueError('expanded partition fingerprint mismatch')
    return retained


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=14)
    parser.add_argument('--device', default='mps')
    # Defaults reproduce the 2026-10-29 expanded run exactly.
    parser.add_argument('--d-model', type=int, default=32)
    parser.add_argument('--modality-dropout', type=float, default=0.)
    parser.add_argument('--selection-metric', choices=('loss', 'family_map', 'family_top1'), default='family_map')
    parser.add_argument('--patience', type=int, default=None)
    parser.add_argument('--seed', type=int, default=44)
    # Memory-only options; they do not change the computed results.
    parser.add_argument('--pre-encode', action='store_true', help='tensorize train/validation once and drop raw events')
    parser.add_argument('--mps-empty-cache-every', type=int, default=None, metavar='N')
    # Optimization variants for A/B runs; defaults reproduce the original run.
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--lr-schedule', choices=('constant', 'warmup_cosine'), default='constant')
    parser.add_argument('--warmup-fraction', type=float, default=.05)
    parser.add_argument('--balanced-sampling', action='store_true')
    # Behavior supervision: rule/weak labels (default, head stays untrained on
    # WinMET) or v5 CAPE behavioral signatures with absent names as negatives.
    parser.add_argument('--behavior-targets', choices=('evidence', 'cape-signatures'), default='evidence')
    parser.add_argument('--behavior-min-support', type=int, default=None)
    args = parser.parse_args()
    if args.checkpoint.exists() or args.report.exists():
        parser.error('refusing to overwrite checkpoint or report')
    if args.epochs < 1:
        parser.error('epochs must be positive')
    manifest = json.loads(args.manifest.read_text())
    if manifest.get('original_split_verified') is not True or manifest.get('old_group_assignments_preserved') is not True:
        parser.error('verified expansion manifest required')
    torch.set_num_threads(2)
    reports = load_expansion(manifest.get('source'), manifest, pack=args.pre_encode)
    result = train_model(
        reports['train'], reports['validation'], args.checkpoint,
        epochs=args.epochs, batch_size=args.batch_size, seed=args.seed, device=args.device, d_model=args.d_model,
        family_min_support=5,
        behavior_min_support=args.behavior_min_support or (200 if args.behavior_targets == 'cape-signatures' else 2),
        calibration_reports=reports['calibration'], test_reports=None,
        train_behavior_head=args.behavior_targets == 'cape-signatures',
        behavior_target_policy='cape_signatures_v1' if args.behavior_targets == 'cape-signatures' else 'evidence_weak_v1',
        family_head_variant='labelwise',
        family_ranking_weight=.5, modality_dropout=args.modality_dropout,
        selection_metric=args.selection_metric, early_stopping_patience=args.patience,
        temporal_window_policy='per_sample_v1', temporal_relation_values=False,
        progress=lambda state: print(json.dumps(state), flush=True),
        pre_encode=args.pre_encode, release_encoded_events=args.pre_encode,
        mps_empty_cache_every=args.mps_empty_cache_every,
        learning_rate=args.lr, lr_schedule=args.lr_schedule,
        warmup_fraction=args.warmup_fraction, balanced_sampling=args.balanced_sampling,
    )
    config = {key: getattr(args, key) for key in ('epochs', 'd_model', 'modality_dropout', 'selection_metric', 'patience', 'seed', 'device',
                                                  'pre_encode', 'mps_empty_cache_every', 'batch_size', 'lr',
                                                  'lr_schedule', 'warmup_fraction', 'balanced_sampling',
                                                  'behavior_targets', 'behavior_min_support')}
    result = {'manifest': manifest, 'config': config, 'result': result, 'test_metrics': None,
              'notice': 'Expanded weak-label training candidate; held-out event objects and predictions were not evaluated. '
                        'Existing archive groups retained; no UI model promotion.'}
    with args.report.open('x') as output:
        json.dump(result, output, indent=2)


if __name__ == '__main__':
    main()
