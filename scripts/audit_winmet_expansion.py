"""Audit train-label support for an expansion preserving every old archive split."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random

from malware_hybrid.schema import ParsedReport
from malware_hybrid.training import research_split

NAMES = ('train', 'validation', 'calibration', 'withheld_internal_test')


def fingerprint(rows):
    return hashlib.sha256('\n'.join(sorted(r.sample_hash for r in rows)).encode()).hexdigest()


def audit(rows, reference):
    seed = int(reference['sample_seed'])
    cap = int(reference['sample_cap'])
    selected = []
    rng = random.Random(seed)
    for index, row in enumerate(rows):
        if index < cap:
            selected.append(row)
        else:
            replacement = rng.randrange(index + 1)
            if replacement < cap:
                selected[replacement] = row
    if len(selected) != cap:
        raise ValueError('source smaller than original reservoir')
    parts = research_split(selected, int(reference['split_seed']))
    if parts[-1] != reference['split_strategy']:
        raise ValueError('split strategy mismatch')
    mapping = {}
    for name, reports in zip(NAMES, parts[:4]):
        if fingerprint(reports) != reference['partitions'][name]['sample_ids_sha256']:
            raise ValueError('original split fingerprint mismatch')
        for report in reports:
            group = report.metadata.get('archive_group_id')
            if not group or (group in mapping and mapping[group] != name):
                raise ValueError('missing or overlapping archive group')
            mapping[group] = name
    expanded = {name: [] for name in NAMES}
    for row in rows:
        group = row.metadata.get('archive_group_id')
        if group not in mapping:
            raise ValueError('new group needs an explicitly assigned partition')
        expanded[mapping[group]].append(row)
    counts = Counter(f for r in expanded['train'] if r.family_supervised for f in r.families)
    old_counts = Counter(f for r in parts[0] if r.family_supervised for f in r.families)
    return {'source_rows': len(rows), 'original_sample_cap': cap,
            'original_split_verified': True, 'old_group_assignments_preserved': True,
            'partitions': {name: {'rows': len(part), 'sample_ids_sha256': fingerprint(part)}
                           for name, part in expanded.items()},
            'train_family_support': dict(sorted(counts.items())),
            'train_support_summary': {str(minimum): {
                'original_families': sum(n >= minimum for n in old_counts.values()),
                'expanded_families': sum(n >= minimum for n in counts.values()),
                'newly_supported_families': sorted(f for f,n in counts.items()
                                                  if n >= minimum and old_counts[f] < minimum)}
                for minimum in (2, 5, 10, 20)},
            'notice': 'Training-label support only. No model trained or held-out predictions evaluated. '
                      'Labels are weak AVClass consensus; expanded partitions retain the old archive-group assignments.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reports', type=Path)
    parser.add_argument('reference', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('refusing to overwrite output')
    reference = json.loads(args.reference.read_text())
    if args.reports.resolve() != Path(reference['source']).resolve():
        parser.error('source differs from reference')
    digest = hashlib.sha256()
    rows = []
    seen = set()
    with args.reports.open('rb') as handle:
        for line in handle:
            digest.update(line)
            data = json.loads(line)
            identity = data['sample_hash']
            if identity in seen:
                raise ValueError('duplicate sample identity')
            seen.add(identity)
            rows.append(ParsedReport(sample_hash=identity, family=data.get('family'),
                       family_labels=data.get('family_labels', []), atomic_evidence=[], events=[],
                       metadata=data.get('metadata', {})))
    if digest.hexdigest() != reference['source_sha256']:
        parser.error('source checksum mismatch')
    result = audit(rows, reference)
    result.update(source_sha256=digest.hexdigest(), source=str(args.reports))
    with args.output.open('x') as target:
        json.dump(result, target, indent=2)
    print(json.dumps(result['train_support_summary']))


if __name__ == '__main__':
    main()
