"""Build a v2 expansion manifest that keeps every v1 partition and the v1 test fixed.

New prepared WinMET JSONL files contribute whole archive groups (solid blocks):
groups listed under ``--second-test-source`` are split by seeded shuffle into a
new ``withheld_test_v2`` partition, extra validation groups and train; groups
from the other new sources go to train only. Rows whose SHA-256 already exists
in an earlier partition or file are excluded (recorded, not deleted). Exact
imphash overlap between new training rows and each withheld partition is
reported for transparency. Only metadata is read; no event objects are built.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random


def scan(path: Path):
    digest, rows = hashlib.sha256(), []
    with path.open('rb') as handle:
        for line in handle:
            digest.update(line)
            payload = json.loads(line)
            metadata = payload.get('metadata', {})
            rows.append({
                'sample_hash': payload['sample_hash'],
                'group': metadata.get('archive_group_id'),
                'families': payload.get('family_labels') or ([payload['family']] if payload.get('family') else []),
                'imphash': (metadata.get('fuzzy_hashes') or {}).get('imphash'),
                'normalization_version': metadata.get('normalization_version'),
            })
    return digest.hexdigest(), rows


def fingerprint(hashes):
    return hashlib.sha256('\n'.join(sorted(hashes)).encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('v1_manifest', type=Path)
    parser.add_argument('--train-source', type=Path, action='append', default=[])
    parser.add_argument('--second-test-source', type=Path, required=True)
    parser.add_argument('--second-test-fraction', type=float, default=.15)
    parser.add_argument('--extra-validation-fraction', type=float, default=.10)
    parser.add_argument('--seed', type=int, default=20261111)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('refusing to overwrite manifest')
    v1 = json.loads(args.v1_manifest.read_text())
    old_path = Path(v1['source'])
    old_sha, old_rows = scan(old_path)
    if old_sha != v1['source_sha256']:
        raise SystemExit('v1 source checksum mismatch')
    group_partition = {group: name for name, part in v1['partitions'].items() for group in part['archive_groups']}
    partitions = {name: {'archive_groups': list(part['archive_groups']), 'hashes': []}
                  for name, part in v1['partitions'].items()}
    partitions['withheld_test_v2'] = {'archive_groups': [], 'hashes': []}
    seen = {}
    imphash_by_partition: dict[str, set] = {name: set() for name in partitions}
    family_support = Counter()
    for row in old_rows:
        name = group_partition[row['group']]
        partitions[name]['hashes'].append(row['sample_hash'])
        seen[row['sample_hash']] = str(old_path)
        if row['imphash']:
            imphash_by_partition[name].add(row['imphash'])
        if name == 'train':
            family_support.update(row['families'])
    for name, part in v1['partitions'].items():
        if fingerprint(partitions[name]['hashes']) != part['sample_ids_sha256']:
            raise SystemExit(f'v1 partition fingerprint mismatch: {name}')

    sources = [{'path': str(old_path), 'sha256': old_sha}]
    excluded, new_train_rows = [], []
    version = Counter(row['normalization_version'] for row in old_rows)
    plan = [(path, False) for path in args.train_source] + [(args.second_test_source, True)]
    for path, split_groups in plan:
        sha, rows = scan(path)
        sources.append({'path': str(path), 'sha256': sha})
        version.update(row['normalization_version'] for row in rows)
        groups = sorted({row['group'] for row in rows})
        collisions = [group for group in groups if group in group_partition]
        if collisions:
            raise SystemExit(f'archive group collision with earlier partitions: {collisions[:3]}')
        assignment = {group: 'train' for group in groups}
        if split_groups:
            shuffled = groups[:]
            random.Random(args.seed).shuffle(shuffled)
            n_test = max(1, round(len(shuffled) * args.second_test_fraction))
            n_val = max(1, round(len(shuffled) * args.extra_validation_fraction))
            for group in shuffled[:n_test]:
                assignment[group] = 'withheld_test_v2'
            for group in shuffled[n_test:n_test + n_val]:
                assignment[group] = 'validation'
        for group, name in assignment.items():
            group_partition[group] = name
            partitions[name]['archive_groups'].append(group)
        for row in rows:
            if row['sample_hash'] in seen:
                excluded.append({'source': str(path), 'sample_hash': row['sample_hash'],
                                 'first_seen_in': seen[row['sample_hash']]})
                continue
            seen[row['sample_hash']] = str(path)
            name = assignment[row['group']]
            partitions[name]['hashes'].append(row['sample_hash'])
            if row['imphash']:
                imphash_by_partition[name].add(row['imphash'])
            if name == 'train':
                family_support.update(row['families'])
                new_train_rows.append(row)
    if set(version) - {'cape-resource-relations-v4'}:
        raise SystemExit(f'mixed normalization versions: {dict(version)}')
    overlap = {}
    for name in ('withheld_internal_test', 'withheld_test_v2'):
        shared = sum(1 for row in new_train_rows if row['imphash'] and row['imphash'] in imphash_by_partition[name])
        overlap[name] = {'new_train_rows_with_imphash_seen_in_partition': shared,
                         'new_train_rows': len(new_train_rows)}
    manifest = {
        'schema': 'winmet_expansion_v2',
        'v1_manifest': str(args.v1_manifest),
        'v1_manifest_sha256': hashlib.sha256(args.v1_manifest.read_bytes()).hexdigest(),
        'original_split_verified': True,
        'old_group_assignments_preserved': True,
        'v1_test_unchanged': partitions['withheld_internal_test']['archive_groups'] == v1['partitions']['withheld_internal_test']['archive_groups'],
        'v1_validation_groups': v1['partitions']['validation']['archive_groups'],
        'sources': sources,
        'excluded_rows': excluded,
        'seed': args.seed,
        'partitions': {name: {'rows': len(part['hashes']), 'sample_ids_sha256': fingerprint(part['hashes']),
                              'archive_groups': part['archive_groups']} for name, part in partitions.items()},
        'train_family_support': dict(sorted(family_support.items())),
        'train_families_support_ge_5': sum(1 for count in family_support.values() if count >= 5),
        'imphash_overlap_new_train_vs_withheld': overlap,
        'notice': 'Weak AVClass consensus labels. v1 partitions and test are unchanged; new archive groups '
                  'go to train, extra validation or withheld_test_v2. No predictions were computed.',
    }
    if not manifest['v1_test_unchanged']:
        raise SystemExit('v1 test changed')
    with args.output.open('x') as handle:
        json.dump(manifest, handle, indent=2)
    print(json.dumps({name: part['rows'] for name, part in manifest['partitions'].items()}
                     | {'excluded_rows': len(excluded), 'families_ge5': manifest['train_families_support_ge_5'],
                        'imphash_overlap': overlap}))


if __name__ == '__main__':
    main()
