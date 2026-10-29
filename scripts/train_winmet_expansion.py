"""Train one expanded Hybrid Attention candidate with frozen archive partitions."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import torch

from malware_hybrid.graph import ExecutionGraphBuilder
from malware_hybrid.serialization import report_from_dict
from malware_hybrid.training import train_model


def load_expansion(source, manifest):
    groups = {}
    for name, part in manifest['partitions'].items():
        for group in part['archive_groups']:
            if group in groups:
                raise ValueError('archive group overlaps partitions')
            groups[group] = name
    retained = {name: [] for name in ('train', 'validation', 'calibration')}
    identities = {name: [] for name in manifest['partitions']}
    seen, digest = set(), hashlib.sha256()
    builder = ExecutionGraphBuilder()
    with Path(source).open('rb') as handle:
        for index, line in enumerate(handle):
            digest.update(line)
            payload = json.loads(line)
            identity = payload['sample_hash']
            if identity in seen:
                raise ValueError('duplicate sample identity')
            seen.add(identity)
            group = payload.get('metadata', {}).get('archive_group_id')
            if group not in groups:
                raise ValueError('unassigned archive group')
            name = groups[group]
            identities[name].append(identity)
            if name == 'withheld_internal_test':
                # Do not construct its event/graph objects or predictions.
                continue
            report = report_from_dict(payload)
            canonical = builder.build(report.events)
            if report.graph is not None and report.graph != canonical:
                raise ValueError('stored graph differs from event reconstruction')
            report.graph = None  # Rebuild on demand; avoids retaining duplicate graph objects.
            retained[name].append(report)
            if (index + 1) % 250 == 0:
                print(json.dumps({'loaded_rows': index + 1}), flush=True)
    if digest.hexdigest() != manifest['source_sha256']:
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
    args = parser.parse_args()
    if args.checkpoint.exists() or args.report.exists():
        parser.error('refusing to overwrite checkpoint or report')
    if args.epochs < 1:
        parser.error('epochs must be positive')
    manifest = json.loads(args.manifest.read_text())
    if manifest.get('original_split_verified') is not True or manifest.get('old_group_assignments_preserved') is not True:
        parser.error('verified expansion manifest required')
    torch.set_num_threads(2)
    reports = load_expansion(manifest['source'], manifest)
    result = train_model(
        reports['train'], reports['validation'], args.checkpoint,
        epochs=args.epochs, batch_size=4, seed=44, device=args.device, d_model=32,
        family_min_support=5, behavior_min_support=2,
        calibration_reports=reports['calibration'], test_reports=None,
        train_behavior_head=False, family_head_variant='labelwise',
        family_ranking_weight=.5, modality_dropout=0., selection_metric='family_map',
        temporal_window_policy='per_sample_v1', temporal_relation_values=False,
        progress=lambda state: print(json.dumps(state), flush=True),
    )
    result = {'manifest': manifest, 'result': result, 'test_metrics': None,
              'notice': 'Expanded weak-label training candidate; held-out event objects and predictions were not evaluated. '
                        'Existing archive groups retained; no UI model promotion.'}
    with args.report.open('x') as output:
        json.dump(result, output, indent=2)


if __name__ == '__main__':
    main()
